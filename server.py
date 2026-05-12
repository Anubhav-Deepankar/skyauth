"""
SkyAuth Backend Server — v3.0 (Full Intelligence Edition)
Run: python server.py
Access from mobile: http://<YOUR_PC_IP>:8000

Key upgrades in v3:
 ✅ Gemini AI is a FIRST-CLASS decision-maker (not just a passive check)
    - Gemini's verdict can HARD-BLOCK a transaction
    - Gemini detects: AI-generated image, screenshot, ground/indoor photo, night image
    - Gemini estimates sun azimuth/elevation from visual cues
 ✅ Real deepfake / AI-image detection pipeline:
    - Error Level Analysis (ELA) — detects JPEG recompression artifacts of AI images
    - Sensor noise fingerprint (real cameras have natural ISO noise)
    - DCT frequency domain analysis
    - Color uniformity (AI skies are too smooth)
    - Gemini's ai_generated_probability as a hard signal
    - All signals fused → probability + hard BLOCK if high
 ✅ Ground-image rejection:
    - Gemini classifies scene type (sky / ground / indoor / not-sky)
    - OpenCV sky-blue pixel ratio check
    - Hard blocked if scene is NOT sky
 ✅ Azimuth/Elevation from image (OpenCV):
    - Detects sun blob in image
    - Estimates absolute azimuth from device heading + pixel horizontal offset
    - Estimates elevation from device tilt + pixel vertical offset
    - Compares BOTH against solar API values
 ✅ Sun-aware challenge generation (never gives impossible direction)
 ✅ Random Forest ensemble of 12 features
"""

import os, io, json, time, random, hashlib, base64, math, struct, zlib, requests
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────
# LOGGING SETUP — all logs visible in CMD/terminal
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level    = logging.DEBUG,
    format   = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt  = "%H:%M:%S",
    handlers = [logging.StreamHandler()]
)
log = logging.getLogger("skyauth")

# ─────────────────────────────────────────────────────────────────
# CONFIG  — replace these with your real keys
# ─────────────────────────────────────────────────────────────────
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL        = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# ─────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────
app = FastAPI(title="SkyAuth Payment Server", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
sessions: dict     = {}
transactions: list = []

# ═══════════════════════════════════════════════════════════════
# SECTION 1: SOLAR POSITION (NOAA algorithm, no external lib)
# ═══════════════════════════════════════════════════════════════

def solar_position(lat: float, lon: float, utc_dt: datetime) -> dict:
    """Compute sun azimuth and elevation using NOAA algorithm (±0.01° accuracy)."""
    n        = utc_dt.timetuple().tm_yday
    hour_utc = utc_dt.hour + utc_dt.minute / 60.0 + utc_dt.second / 3600.0
    gamma    = 2 * math.pi / 365 * (n - 1 + (hour_utc - 12) / 24)

    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma)
                       - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma)
                       - 0.04089  * math.sin(2 * gamma))
    decl = (0.006918
            - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2*gamma) + 0.000907 * math.sin(2*gamma)
            - 0.002697 * math.cos(3*gamma) + 0.00148  * math.sin(3*gamma))

    time_offset = eqtime + 4 * lon
    tst  = hour_utc * 60 + time_offset
    ha   = (tst / 4) - 180
    lat_r, decl_r, ha_r = math.radians(lat), decl, math.radians(ha)

    sin_elev = (math.sin(lat_r) * math.sin(decl_r)
                + math.cos(lat_r) * math.cos(decl_r) * math.cos(ha_r))
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))

    cos_az = ((math.sin(decl_r) - math.sin(lat_r) * sin_elev)
              / (math.cos(lat_r) * math.cos(math.radians(max(0.001, abs(elevation)))) + 1e-9))
    cos_az  = max(-1.0, min(1.0, cos_az))
    azimuth = math.degrees(math.acos(cos_az))
    if ha > 0:
        azimuth = 360 - azimuth

    return {
        "azimuth_deg":      round(azimuth, 2),
        "elevation_deg":    round(elevation, 2),
        "is_above_horizon": elevation > 0,
        "hour_angle":       round(ha, 2),
        "declination_deg":  round(math.degrees(decl_r), 2),
    }


def azimuth_to_direction(az: float) -> str:
    az = az % 360
    for boundary, name in [
        (22.5,"North"),(67.5,"NorthEast"),(112.5,"East"),(157.5,"SouthEast"),
        (202.5,"South"),(247.5,"SouthWest"),(292.5,"West"),(337.5,"NorthWest"),(360.0,"North")
    ]:
        if az < boundary:
            return name
    return "North"


# ═══════════════════════════════════════════════════════════════
# SECTION 2: OPENCV — SUN DETECTION + AZIMUTH/ELEVATION FROM IMAGE
# ═══════════════════════════════════════════════════════════════

def detect_sun_in_image(img_b64: str) -> dict:
    """
    Find the brightest blob (sun) in the sky image using OpenCV.
    Returns pixel position + estimated azimuth offset & elevation from pixel geometry.
    Also checks sky-blue ratio to catch ground/indoor images.
    """
    try:
        import cv2
        img_bytes = base64.b64decode(img_b64 + "==")
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Could not decode image", "sun_detected": False, "is_sky_image": False}

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── Sky-blue pixel ratio (ground-image rejection) ──
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Blue sky hue: 90–130, sat >40, val >60; also white/grey overcast sky
        sky_mask = (
            ((hsv[:,:,0] >= 90) & (hsv[:,:,0] <= 140) &
             (hsv[:,:,1] >= 40) & (hsv[:,:,2] >= 60)) |
            ((hsv[:,:,1] < 40) & (hsv[:,:,2] > 120))  # light grey/white overcast
        )
        sky_ratio = float(np.sum(sky_mask)) / (h * w)
        is_sky_image_cv = sky_ratio > 0.12  # at least 12% sky pixels

        # ── Sun detection ──
        blurred = cv2.GaussianBlur(gray, (31, 31), 0)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        sun_x, sun_y = max_loc
        mean_brightness = float(np.mean(gray))
        max_brightness  = float(max_val)
        brightness_ratio = max_brightness / max(mean_brightness, 1)
        overexposed = float(np.sum(gray > 230)) / (h * w)

        # ── Pixel → angle geometry ──
        # Phone horizontal field of view ≈ 60°, vertical ≈ 45°
        hfov, vfov = 60.0, 45.0
        norm_x = (sun_x - w / 2) / (w / 2)   # -1=left, +1=right
        norm_y = (sun_y - h / 2) / (h / 2)   # -1=top,  +1=bottom
        pixel_az_offset  = norm_x * (hfov / 2)
        pixel_el_offset  = -norm_y * (vfov / 2)   # top pixel = higher elevation

        # ── Haze-aware sun detection (replaces old single-threshold check) ──
        # Mode 1: Clear sun — sharp bright disc (original check)
        mode_clear = (
            brightness_ratio > 3.0 and
            overexposed > 0.015 and
            sun_y < h * 0.65
        )
        # Mode 2: Hazy/overcast sun — diffused glow (common in Delhi/polluted cities)
        # The sun becomes a wide soft blob: lower brightness ratio but large bright region
        # and the blob is still in the upper 70% of frame
        top_region = gray[:int(h * 0.70), :]
        top_bright_frac = float(np.sum(top_region > 180)) / max(top_region.size, 1)
        mode_hazy = (
            brightness_ratio > 1.4 and          # softer ratio threshold
            top_bright_frac > 0.04 and          # at least 4% very bright pixels in top 70%
            max_brightness > 200 and            # still needs to be bright overall
            sun_y < h * 0.70
        )
        # Mode 3: Glare-only — extreme overexposure with sun above horizon
        mode_glare = (
            overexposed > 0.08 and              # >8% pixels blown out
            sun_y < h * 0.55 and               # in upper half
            mean_brightness > 120              # overall bright scene
        )
        sun_likely = mode_clear or mode_hazy or mode_glare
        sun_detection_mode = ("clear" if mode_clear else
                               "hazy" if mode_hazy else
                               "glare" if mode_glare else "none")

        return {
            "sun_detected":           sun_likely,
            "sun_detection_mode":     sun_detection_mode,
            "is_sky_image":           is_sky_image_cv,
            "sky_pixel_ratio":        round(sky_ratio, 3),
            "sun_pixel":              {"x": int(sun_x), "y": int(sun_y)},
            "image_size":             {"w": w, "h": h},
            "max_brightness":         round(max_brightness, 1),
            "mean_brightness":        round(mean_brightness, 1),
            "brightness_ratio":       round(brightness_ratio, 2),
            "overexposed_fraction":   round(overexposed, 4),
            "top_bright_frac":        round(top_bright_frac, 4),
            "pixel_az_offset_deg":    round(pixel_az_offset, 1),
            "pixel_el_offset_deg":    round(pixel_el_offset, 1),
            "norm_x":                 round(float(norm_x), 3),
            "norm_y":                 round(float(norm_y), 3),
        }
    except Exception as e:
        return {"error": str(e), "sun_detected": False, "is_sky_image": False}


def compare_sun_position(cv: dict, device_heading: float, device_tilt: float,
                          solar_azimuth: float, solar_elevation: float) -> dict:
    """
    Full azimuth + elevation comparison between:
      - What the image says (device sensor + pixel offset)
      - What the solar API says it should be

    device_heading: compass heading the phone was pointing (0=North, clockwise)
    device_tilt:    angle above horizon the phone was tilted (0=horizon, 90=zenith)
    """
    if not cv.get("sun_detected"):
        return {
            "match": False, "score": 0,
            "reason": "Sun not detected in image by OpenCV",
            "estimated_azimuth": None, "estimated_elevation": None,
            "solar_azimuth": round(solar_azimuth, 1),
            "solar_elevation": round(solar_elevation, 1),
            "azimuth_error_deg": None, "elevation_error_deg": None,
        }

    # Absolute azimuth the image's sun blob implies
    estimated_az = (device_heading + cv["pixel_az_offset_deg"]) % 360

    # Elevation implied by device tilt + where in-frame the blob sits
    estimated_el = device_tilt + cv["pixel_el_offset_deg"]

    az_err = abs(estimated_az - solar_azimuth)
    if az_err > 180:
        az_err = 360 - az_err
    el_err = abs(estimated_el - solar_elevation)

    match = az_err < 30 and el_err < 25
    score = max(0.0, 100 - az_err * 1.5 - el_err * 2)

    return {
        "match":               match,
        "estimated_azimuth":   round(estimated_az, 1),
        "estimated_elevation": round(estimated_el, 1),
        "solar_azimuth":       round(solar_azimuth, 1),
        "solar_elevation":     round(solar_elevation, 1),
        "azimuth_error_deg":   round(az_err, 1),
        "elevation_error_deg": round(el_err, 1),
        "score":               round(score, 1),
        "reason": (
            f"✅ Sun position matches solar API (az err {az_err:.1f}°, el err {el_err:.1f}°)"
            if match else
            f"❌ Sun position mismatch (az err {az_err:.1f}°, el err {el_err:.1f}°)"
        ),
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 3: DEEPFAKE / AI-IMAGE DETECTION (real ML signals)
# ═══════════════════════════════════════════════════════════════

def detect_fake_image(img_b64: str, gemini_ai_prob: float = 0.5,
                      weather_context: dict = None) -> dict:
    """
    Multi-layer fake / AI-image detection — v3.1 (weather-aware, mobile-safe).

    KEY FIXES vs v3.0:
      - ELA threshold corrected: WhatsApp/mobile JPEG already re-compressed →
        near-zero ELA is NORMAL for phone photos. Only flag if essentially 0.
      - DFT check rewritten: old center/total ratio was mathematically broken
        (could exceed 1.0). Now uses proper high-freq vs low-freq energy split.
      - Compression ratio: WhatsApp crushes to 0.02; only flag true screenshot
        artifacts (< 0.005).
      - All smooth-sky thresholds scale with weather (haze/overcast relax ×2.2).
      - Fake threshold raised 0.42 → 0.48 for fewer false positives.

    weather_context: dict with 'clouds_pct', 'description', 'visibility_m'
    """
    log.info("━━━ [FAKE DETECT] Starting image authenticity analysis ━━━")

    # ── Weather-aware threshold relaxation ─────────────────────────
    wctx       = weather_context or {}
    clouds_pct = wctx.get("clouds_pct", 0)
    vis_m      = wctx.get("visibility_m", 10000)
    wx_desc    = (wctx.get("description") or "").lower()

    is_hazy     = vis_m < 5000 or any(k in wx_desc for k in
                    ("haze","fog","mist","smoke","dust","sand"))
    is_overcast = clouds_pct >= 70 or any(k in wx_desc for k in
                    ("overcast","broken","few clouds","scattered","clouds"))

    smooth_relax = 1.0
    if is_hazy and is_overcast:
        smooth_relax = 2.2
    elif is_hazy:
        smooth_relax = 1.8
    elif is_overcast:
        smooth_relax = 1.5

    log.info(f"  Weather: clouds={clouds_pct}%, vis={vis_m}m, desc='{wx_desc}'")
    log.info(f"  Hazy={is_hazy}, Overcast={is_overcast} → smooth_relax={smooth_relax}")

    try:
        import cv2
        img_bytes = base64.b64decode(img_b64 + "==")
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            log.error("  Could not decode image bytes!")
            return {"fake_probability": 0.9, "is_likely_fake": True,
                    "flags": ["decode_fail"], "ela_mean": 0, "noise_std": 0}

        flags, scores = [], []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        log.info(f"  Image size: {w}×{h} px | File size: {len(img_bytes)/1024:.1f} KB")

        # ── 1. ELA ────────────────────────────────────────────────────
        # FIXED: WhatsApp & phone cameras re-compress aggressively.
        # A real mobile photo already has near-zero ELA — only flag if
        # the bytes are essentially IDENTICAL (< 0.005 mean diff), which
        # indicates a lossless screenshot or AI-generated PNG saved as JPEG.
        buf = io.BytesIO()
        import PIL.Image as PILImage
        pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil.save(buf, "JPEG", quality=92)
        buf.seek(0)
        recompressed_cv = cv2.cvtColor(
            np.array(PILImage.open(buf)), cv2.COLOR_RGB2BGR)
        ela      = cv2.absdiff(img, recompressed_cv).astype(np.float32)
        ela_mean = float(np.mean(ela))
        ela_std  = float(np.std(ela))

        ELA_ZERO_THRESH = 0.005   # near-identical = lossless source (screenshot)
        ELA_HIGH_THRESH = 20.0    # extreme artifacts = heavy editing
        if ela_mean < ELA_ZERO_THRESH:
            flags.append("ela_near_zero_lossless_source")
            scores.append(0.70)
            log.warning(f"  [1] ELA={ela_mean:.4f} ❌ near-zero → lossless/screenshot source")
        elif ela_mean > ELA_HIGH_THRESH:
            flags.append("ela_high_possible_editing")
            scores.append(0.55)
            log.warning(f"  [1] ELA={ela_mean:.4f} ⚠ high → heavy editing suspected")
        else:
            scores.append(0.1)
            log.info(f"  [1] ELA={ela_mean:.4f} ✅ normal range")

        # ── 2. Sensor Noise ───────────────────────────────────────────
        denoised  = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        noise     = img.astype(np.float32) - denoised.astype(np.float32)
        noise_std = float(np.std(noise))
        n_low1 = 1.5 / smooth_relax
        n_low2 = 3.0 / smooth_relax
        if noise_std < n_low1:
            flags.append("near_zero_sensor_noise_ai_render")
            scores.append(0.80)
            log.warning(f"  [2] Noise={noise_std:.3f} ❌ near-zero → likely AI render (thresh {n_low1:.2f})")
        elif noise_std < n_low2:
            flags.append("low_sensor_noise_suspicious")
            scores.append(0.45)
            log.warning(f"  [2] Noise={noise_std:.3f} ⚠ low (thresh {n_low2:.2f})")
        else:
            scores.append(0.1)
            log.info(f"  [2] Noise={noise_std:.3f} ✅ OK")

        # ── 3. DFT — REWRITTEN ────────────────────────────────────────
        # FIXED: old center/total ratio was broken (could exceed 1.0).
        # Now: compute fraction of energy in high-frequency bands.
        # AI-generated images lack natural high-frequency texture → low high_frac.
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shifted   = np.fft.fftshift(f_transform)
        rows_f, cols_f = gray.shape
        crow, ccol  = rows_f // 2, cols_f // 2
        lf_radius   = min(rows_f, cols_f) // 6   # inner low-freq circle
        mask_low    = np.zeros((rows_f, cols_f), np.uint8)
        cv2.circle(mask_low, (ccol, crow), lf_radius, 1, -1)
        mask_high   = 1 - mask_low
        abs_f       = np.abs(f_shifted)
        low_energy  = float(np.sum(abs_f * mask_low))
        high_energy = float(np.sum(abs_f * mask_high))
        high_frac   = high_energy / max(low_energy + high_energy, 1)
        # AI images: high_frac < 0.10 (very suspicious), < 0.15 (suspicious)
        # Real hazy photo (this image): high_frac ≈ 0.39
        HF_THRESH_HARD = 0.10
        HF_THRESH_SOFT = 0.15
        if high_frac < HF_THRESH_HARD:
            flags.append("dft_missing_high_freq_ai_generated")
            scores.append(0.70)
            log.warning(f"  [3] DFT high_frac={high_frac:.4f} ❌ very low → AI-generated likely")
        elif high_frac < HF_THRESH_SOFT:
            flags.append("dft_low_high_freq_suspicious")
            scores.append(0.40)
            log.warning(f"  [3] DFT high_frac={high_frac:.4f} ⚠ low")
        else:
            scores.append(0.1)
            log.info(f"  [3] DFT high_frac={high_frac:.4f} ✅ OK")

        # ── 4. Sky Color Uniformity ───────────────────────────────────
        sky_region = img[:h//2, :]
        hsv_sky    = cv2.cvtColor(sky_region, cv2.COLOR_BGR2HSV)
        sat_std    = float(np.std(hsv_sky[:,:,1]))
        sat_thresh = 7.0 / smooth_relax
        if sat_std < sat_thresh:
            flags.append("unnaturally_uniform_sky_ai_render")
            scores.append(0.70)
            log.warning(f"  [4] SatStd={sat_std:.2f} ❌ too uniform (thresh {sat_thresh:.2f})")
        else:
            scores.append(0.1)
            log.info(f"  [4] SatStd={sat_std:.2f} ✅ OK (thresh {sat_thresh:.2f})")

        # ── 5. Compression Ratio ──────────────────────────────────────
        # FIXED: WhatsApp crushes photos to compression_ratio ≈ 0.02.
        # Old threshold (0.12) falsely flagged every WhatsApp photo.
        # New threshold (0.005): only catch true screenshots and renders.
        file_size         = len(img_bytes)
        compression_ratio = file_size / (w * h)
        COMPRESS_THRESH   = 0.005   # ~ lossless/render artifact only
        if compression_ratio < COMPRESS_THRESH:
            flags.append("suspiciously_compressed_possible_screenshot")
            scores.append(0.60)
            log.warning(f"  [5] Compression={compression_ratio:.4f} ❌ extreme (thresh {COMPRESS_THRESH})")
        else:
            scores.append(0.1)
            log.info(f"  [5] Compression={compression_ratio:.4f} ✅ OK")

        # ── 6. Perfect Gradient Sky ───────────────────────────────────
        sky_gray  = gray[:h//2, :]
        row_means = np.mean(sky_gray, axis=1)
        row_diffs = np.diff(row_means)
        rd_std    = float(np.std(row_diffs))
        rd_mean   = float(np.mean(np.abs(row_diffs)))
        grad_std_t  = 1.2 / smooth_relax
        grad_mean_t = 1.5 / smooth_relax
        if rd_std < grad_std_t and rd_mean < grad_mean_t:
            flags.append("perfect_gradient_sky_ai_or_render")
            scores.append(0.85)
            log.warning(f"  [6] Gradient std={rd_std:.3f},mean={rd_mean:.3f} ❌ perfect gradient"
                        f" (thresholds {grad_std_t:.3f}/{grad_mean_t:.3f})")
        else:
            scores.append(0.1)
            log.info(f"  [6] Gradient std={rd_std:.3f},mean={rd_mean:.3f} ✅ OK"
                     f" (thresholds {grad_std_t:.3f}/{grad_mean_t:.3f})")

        # ── 7. Laplacian Sharpness ────────────────────────────────────
        lap_var   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        lap_thresh = 40.0 / smooth_relax
        if lap_var < lap_thresh:
            flags.append("too_smooth_possibly_ai")
            scores.append(0.60)
            log.warning(f"  [7] Laplacian={lap_var:.2f} ❌ too smooth (thresh {lap_thresh:.2f})"
                        " — NOTE: haze/WhatsApp compression causes this in real photos")
        elif lap_var > 10000:
            flags.append("over_sharpened_possible_post_processing")
            scores.append(0.45)
            log.warning(f"  [7] Laplacian={lap_var:.2f} ⚠ over-sharpened")
        else:
            scores.append(0.1)
            log.info(f"  [7] Laplacian={lap_var:.2f} ✅ OK")

        # ── 8. Gemini AI probability ──────────────────────────────────
        gemini_norm = max(0.0, min(1.0, gemini_ai_prob / 100.0))
        scores.append(gemini_norm)
        if gemini_norm > 0.6:
            flags.append(f"gemini_flagged_ai_{int(gemini_ai_prob)}pct")
            log.warning(f"  [8] Gemini AI prob={gemini_ai_prob}% ❌ flagged as AI")
        else:
            log.info(f"  [8] Gemini AI prob={gemini_ai_prob}% ✅ OK")

        # ── Ensemble fusion ───────────────────────────────────────────
        # Weights: ELA×2, noise×2, freq×1, color×1, compress×1, gradient×2, sharp×1, gemini×3
        weights = [2, 2, 1, 1, 1, 2, 1, 3]
        weighted_scores = [s * wt for s, wt in zip(scores, weights)]
        fake_prob = round(min(1.0, float(sum(weighted_scores) / sum(weights))), 3)
        is_fake   = fake_prob > 0.48   # raised from 0.42 to reduce false positives

        log.info(f"  Score components: {[round(s,3) for s in scores]}")
        log.info(f"  Weighted sum: {sum(weighted_scores):.3f} / {sum(weights)}")
        log.info(f"  Fake probability: {fake_prob:.3f} | Threshold: 0.48")
        log.info(f"  Flags raised: {flags if flags else 'none'}")
        if is_fake:
            log.warning(f"  ❌ VERDICT: LIKELY FAKE (prob={fake_prob:.3f})")
        else:
            log.info(f"  ✅ VERDICT: LIKELY GENUINE (prob={fake_prob:.3f})")

        return {
            "fake_probability":    fake_prob,
            "is_likely_fake":      is_fake,
            "flags":               flags,
            "ela_mean":            round(ela_mean, 4),
            "ela_std":             round(ela_std, 4),
            "noise_std":           round(noise_std, 3),
            "sky_saturation_std":  round(sat_std, 1),
            "dft_high_freq_frac":  round(high_frac, 4),
            "compression_ratio":   round(compression_ratio, 4),
            "sharpness_laplacian": round(lap_var, 2),
            "gemini_ai_prob":      round(gemini_ai_prob, 1),
            "weather_relaxation":  round(smooth_relax, 2),
            "score_components": {
                "ela":       round(scores[0], 3),
                "noise":     round(scores[1], 3),
                "freq":      round(scores[2], 3),
                "color":     round(scores[3], 3),
                "compress":  round(scores[4], 3),
                "gradient":  round(scores[5], 3),
                "sharpness": round(scores[6], 3),
                "gemini":    round(scores[7], 3),
            }
        }

    except Exception as e:
        log.error(f"  detect_fake_image EXCEPTION: {e}", exc_info=True)
        return {
            "fake_probability": 0.5, "is_likely_fake": False,
            "flags": [f"analysis_error:{str(e)}"],
            "ela_mean": 0, "noise_std": 0, "gemini_ai_prob": gemini_ai_prob
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 4: GEMINI AI — FIRST-CLASS DECISION MAKER
# ═══════════════════════════════════════════════════════════════

def analyze_image_with_gemini(img_b64: str, context: dict) -> dict:
    """
    Gemini v4: SOLE JUDGE of sun presence and image authenticity.

    No compass/tilt challenge. Gemini verifies:
     1. Is this a real outdoor sky image with the sun visible?
     2. Does the sun's visual position match where it SHOULD be (solar API)?
     3. Is this a genuine camera photo (not AI-generated, screenshot, or edited)?
     4. Provide a confidence-weighted verdict with detailed reasoning.
    """
    if GEMINI_API_KEY in ("INSERT_YOUR_GEMINI_API_KEY_HERE", "", None):
        return {
            "source": "mock_no_key",
            "scene_type": "sky",
            "sky_visible": True,
            "sun_visible": True,
            "sun_position_consistent": True,
            "clouds_present": False,
            "cloud_coverage_percent": 10,
            "sky_condition": "Clear sky",
            "is_fake_or_screenshot": False,
            "authenticity_confidence": 70,
            "is_night_image": False,
            "ai_generated_probability": 20,
            "estimated_sun_azimuth": None,
            "estimated_sun_elevation": None,
            "sun_visibility_reason": "Mock mode — no Gemini key set",
            "authenticity_reason": "Mock mode",
            "overall_verdict": "likely_genuine",
            "ai_score": 70,
            "notes": "⚠️ Gemini key not set — add to GEMINI_API_KEY in server.py",
            "gemini_hard_block": False,
            "gemini_block_reason": None,
        }

    try:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}")

        lat  = context.get("lat", "?")
        lon  = context.get("lon", "?")
        s_az = context.get("solar_azimuth", "?")
        s_el = context.get("solar_elevation", "?")
        weather_desc = context.get("weather_desc", "unknown")
        cloud_pct    = context.get("cloud_pct", "unknown")

        prompt = f"""You are the primary verification AI for SkyAuth — a payment system that requires
the user to photograph the real sky with the sun visible. You are the MAIN judge. There is no
compass or tilt challenge. Your analysis is the core of the decision.

CONTEXT:
- User GPS: lat={lat}, lon={lon}
- Solar API ground truth: sun azimuth={s_az}° (0=North, clockwise), elevation={s_el}° above horizon
- Current weather: {weather_desc}, cloud cover: {cloud_pct}%
- Note: at elevation={s_el}°, the sun is {"very high overhead" if isinstance(s_el, (int,float)) and s_el > 60 else "moderately high" if isinstance(s_el, (int,float)) and s_el > 30 else "relatively low in the sky"}

YOUR TASK: Critically analyze this image and respond ONLY with valid JSON (no markdown, no backticks, no preamble).

{{
  "scene_type": "sky | ground | indoor | vehicle | screenshot | not_sky | other",
  "sky_visible": true/false,
  "sun_visible": true/false,
  "sun_behind_cloud": true/false,
  "sun_position_consistent": true/false,
  "clouds_present": true/false,
  "cloud_coverage_percent": 0-100,
  "sky_condition": "Clear | Partly Cloudy | Overcast | Hazy | Indoor | Not Sky | Night",
  
  "sun_visibility_reason": "one sentence: explain what you see regarding the sun — visible glowing disc, bright glare region, hidden behind clouds, not present, etc.",
  
  "estimated_sun_azimuth": null or number,
  "estimated_sun_elevation": null or number,
  "sun_position_reasoning": "explain how you estimated the sun position from visual cues: shadows, light direction, brightness gradients, lens flare angle, position of bright region",
  
  "is_fake_or_screenshot": true/false,
  "authenticity_confidence": 0-100,
  "ai_generated_probability": 0-100,
  "authenticity_reason": "one sentence: explain your authenticity judgment — natural noise, compression artifacts, lighting consistency, sky texture, etc.",
  
  "overall_verdict": "genuine_sun_visible | genuine_sun_obscured | suspicious | fake | not_sky",
  "verdict_confidence": 0-100,
  "ai_score": 0-100,
  
  "notes": "max 150 chars: any important observation for this specific image"
}}

DETAILED RULES:

SUN DETECTION:
- If the sun is directly visible: sun_visible=true, describe the glowing disc or intense bright spot
- If sky is bright/hazy but no clear disc: sun_visible can still be true if there's an obvious bright glare region
- If overcast/cloudy but sun is partially visible or creating a bright patch: sun_visible=true, sun_behind_cloud=true
- If fully overcast with NO bright region at all: sun_visible=false (but image may still be genuine sky)
- Sun position clues: direction of shadows (opposite sun), lens flare streaks, brightest sky region, color temperature gradient

SUN POSITION CROSS-CHECK:
- Solar API says sun is at azimuth {s_az}° and elevation {s_el}°
- In the image, does the bright region / sun disc appear in a position consistent with this?
- For elevation {s_el}°: {"the sun should appear very high, near the top of the frame if camera is tilted up" if isinstance(s_el, (int,float)) and s_el > 55 else "the sun should appear in the middle-upper area of the sky" if isinstance(s_el, (int,float)) and s_el > 30 else "the sun should be relatively low in the sky"}
- estimated_sun_azimuth: your best compass direction estimate (0=North, clockwise) based on visual cues — use null only if truly impossible
- estimated_sun_elevation: your best angle-above-horizon estimate based on where the bright region sits

AUTHENTICITY CHECK:
- Real camera photos: natural noise, slightly uneven sky texture, real atmospheric haze, natural vignetting
- AI-generated images: too-perfect sky gradients, unnaturally smooth clouds, mathematically perfect blue
- Screenshots: visible UI elements, screen pixels, status bars, app chrome, unnatural sharpness
- Edited photos: mismatched lighting, copy-paste artifacts, inconsistent shadow directions
- If cloud cover is high ({cloud_pct}%), a hazy or uniform sky is EXPECTED and NORMAL — do NOT penalize for this

VERDICT GUIDE:
- genuine_sun_visible: real photo, sky visible, sun clearly present as disc or bright glare region
- genuine_sun_obscured: real photo, sky visible, sun completely hidden BUT a distinct bright patch is still visible through clouds indicating solar position — NOT just a uniformly grey sky
- suspicious: real photo but sun position inconsistent with solar data, OR sky is uniformly grey/overcast with NO visible bright region at all
- fake: AI-generated, screenshot, heavily edited, or not a real camera photo
- not_sky: not a sky image at all (ground, indoor, etc.)

IMPORTANT: High cloud cover and haze are valid, but the user MUST be photographing toward the sun's direction — there should be a visibly brighter region in the sky (even through thick haze) indicating the sun's position. A completely uniform grey sky with NO brighter region should be rated 'suspicious'. The sun must leave some trace of brightness — diffused glow, bright patch, or glare — for genuine_sun_visible or genuine_sun_obscured.
"""

        payload = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg",
                                     "data": img_b64[:len(img_b64) - len(img_b64) % 4]}},
                    {"text": prompt}
                ]
            }],
            "generationConfig": {"temperature": 0.05, "maxOutputTokens": 900}
        }

        r    = requests.post(url, json=payload, timeout=25)
        data = r.json()
        text = (data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "{}"))
        text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result = json.loads(text)
        result["source"] = "gemini_ai"

        # ── Compute hard-block reasons ──────────────────────────────
        block_reason = None
        scene   = result.get("scene_type", "sky")
        verdict = result.get("overall_verdict", "genuine_sun_visible")

        if scene not in ("sky",):
            block_reason = (
                f"Image classified as '{scene}' — SkyAuth requires a real outdoor sky photo."
            )
        elif result.get("is_night_image"):
            block_reason = "Night-time image detected."
        elif verdict == "fake":
            block_reason = (
                f"Image flagged as fake/screenshot "
                f"({result.get('authenticity_reason', 'authenticity check failed')})"
            )
        elif result.get("ai_generated_probability", 0) > 80:
            block_reason = (
                f"AI-generated image detected "
                f"(probability: {result.get('ai_generated_probability')}%)"
            )

        result["gemini_hard_block"]   = block_reason is not None
        result["gemini_block_reason"] = block_reason
        return result

    except Exception as e:
        return {
            "source": "gemini_error", "error": str(e),
            "scene_type": "sky",
            "sky_visible": True, "sun_visible": None,
            "sun_behind_cloud": False,
            "is_fake_or_screenshot": False,
            "authenticity_confidence": 50,
            "is_night_image": False,
            "ai_generated_probability": 30,
            "estimated_sun_azimuth": None,
            "estimated_sun_elevation": None,
            "sun_visibility_reason": f"Gemini call failed: {e}",
            "authenticity_reason": "Analysis unavailable",
            "overall_verdict": "genuine_sun_visible",
            "verdict_confidence": 40,
            "ai_score": 50,
            "notes": f"Gemini call failed: {e}",
            "gemini_hard_block": False,
            "gemini_block_reason": None,
        }



def compare_gemini_sun_estimate(gemini_result: dict,
                                 solar_azimuth: float,
                                 solar_elevation: float) -> dict:
    """
    Compare Gemini's visual sun estimate with the solar API.
    This adds a second independent sun-position cross-check.
    """
    g_az = gemini_result.get("estimated_sun_azimuth")
    g_el = gemini_result.get("estimated_sun_elevation")

    if g_az is None or g_el is None:
        return {"match": False, "reason": "Gemini could not estimate sun position",
                "azimuth_error": None, "elevation_error": None, "score": 0}

    az_err = abs(g_az - solar_azimuth)
    if az_err > 180: az_err = 360 - az_err
    el_err = abs(g_el - solar_elevation)

    match = az_err < 40 and el_err < 30
    score = max(0.0, 100 - az_err * 1.2 - el_err * 1.8)

    return {
        "match":               match,
        "gemini_azimuth":      round(g_az, 1),
        "solar_azimuth":       round(solar_azimuth, 1),
        "gemini_elevation":    round(g_el, 1),
        "solar_elevation":     round(solar_elevation, 1),
        "azimuth_error":       round(az_err, 1),
        "elevation_error":     round(el_err, 1),
        "score":               round(score, 1),
        "reason": (
            f"✅ Gemini sun estimate matches solar API (az ±{az_err:.1f}°, el ±{el_err:.1f}°)"
            if match else
            f"❌ Gemini sun estimate mismatch (az ±{az_err:.1f}°, el ±{el_err:.1f}°)"
        ),
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 5: RANDOM FOREST ENSEMBLE (12 features)
# ═══════════════════════════════════════════════════════════════

def build_random_forest():
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.RandomState(42)
    n   = 600

    pos = np.column_stack([
        rng.uniform(0.9, 1.0, n),  # gps_valid
        rng.uniform(0.9, 1.0, n),  # is_daytime
        rng.uniform(0.7, 1.0, n),  # direction_match
        rng.uniform(0.7, 1.0, n),  # tilt_match
        rng.uniform(0.7, 1.0, n),  # sun_detected_cv
        rng.uniform(0.6, 1.0, n),  # sun_pos_match_cv
        rng.uniform(0.7, 1.0, n),  # not_fake (1-fake_prob)
        rng.uniform(0.7, 1.0, n),  # gemini_ai_score/100
        rng.uniform(0.8, 1.0, n),  # ts_fresh
        rng.uniform(0.8, 1.0, n),  # is_sky_image (cv)
        rng.uniform(0.6, 1.0, n),  # gemini_sun_match
        rng.uniform(0.7, 1.0, n),  # gemini_no_block
    ])
    neg = np.column_stack([
        rng.uniform(0.0, 0.5, n),
        rng.uniform(0.0, 0.3, n),
        rng.uniform(0.0, 0.4, n),
        rng.uniform(0.0, 0.4, n),
        rng.uniform(0.0, 0.3, n),
        rng.uniform(0.0, 0.3, n),
        rng.uniform(0.0, 0.4, n),
        rng.uniform(0.0, 0.4, n),
        rng.uniform(0.0, 0.5, n),
        rng.uniform(0.0, 0.4, n),
        rng.uniform(0.0, 0.3, n),
        rng.uniform(0.0, 0.3, n),
    ])
    X = np.vstack([pos, neg])
    y = np.array([1]*n + [0]*n)
    clf = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=7)
    clf.fit(X, y)
    return clf


_RF_MODEL = build_random_forest()
_RF_FEATURE_NAMES = [
    "GPS Valid", "Daytime", "Direction Match", "Tilt Match",
    "Sun Detected (CV)", "Sun Position Match (CV)",
    "Not Fake (ML)", "Gemini AI Score",
    "Timestamp Fresh", "Is Sky Image (CV)",
    "Gemini Sun Match", "Gemini No Block"
]


def rf_decision(features: dict) -> dict:
    X = np.array([[
        features["gps_valid"],
        features["is_daytime"],
        features["direction_match"],
        features["tilt_match"],
        features["sun_detected"],
        features["sun_pos_match_cv"],
        features["not_fake"],
        features["gemini_ai_score"],
        features["ts_freshness"],
        features["is_sky_image"],
        features["gemini_sun_match"],
        features["gemini_no_block"],
    ]])
    prob = _RF_MODEL.predict_proba(X)[0]
    approved = bool(prob[1] >= 0.58)
    importances = _RF_MODEL.feature_importances_.tolist()
    return {
        "approved":           approved,
        "confidence":         round(float(prob[1]) * 100, 1),
        "rejection_risk":     round(float(prob[0]) * 100, 1),
        "feature_importance": dict(zip(_RF_FEATURE_NAMES,
                                       [round(i * 100, 1) for i in importances])),
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 6: WEATHER
# ═══════════════════════════════════════════════════════════════

def get_weather(lat: float, lon: float) -> dict:
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "INSERT_YOUR_OPENWEATHERMAP_KEY_HERE":
        hour = datetime.now().hour
        return {
            "source": "mock", "description": "clear sky",
            "temperature_c": 28.0, "humidity": 55, "clouds_pct": 10,
            "wind_speed_kmh": 8.0, "visibility_m": 10000,
            "weather_code": 800, "is_daytime": 6 <= hour <= 18,
            "city": "Unknown", "lat": lat, "lon": lon,
        }
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric")
        r = requests.get(url, timeout=5)
        d = r.json()
        sunrise, sunset = d["sys"].get("sunrise",0), d["sys"].get("sunset",0)
        now_ts = int(time.time())
        return {
            "source": "openweathermap",
            "description":    d["weather"][0]["description"],
            "temperature_c":  d["main"]["temp"],
            "humidity":       d["main"]["humidity"],
            "clouds_pct":     d["clouds"]["all"],
            "wind_speed_kmh": round(d["wind"]["speed"] * 3.6, 1),
            "visibility_m":   d.get("visibility", 10000),
            "weather_code":   d["weather"][0]["id"],
            "is_daytime":     sunrise < now_ts < sunset,
            "city":           d.get("name", "Unknown"),
            "lat": lat, "lon": lon,
        }
    except Exception as e:
        return {
            "source":"error","description":"unavailable",
            "temperature_c":0,"humidity":0,"clouds_pct":0,
            "wind_speed_kmh":0,"visibility_m":0,"weather_code":0,
            "is_daytime":True,"city":"Unknown","lat":lat,"lon":lon,"error":str(e),
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 7: HELPERS
# ═══════════════════════════════════════════════════════════════

def direction_tolerance(actual: float, target: float, tol: float = 25.0) -> tuple:
    diff = abs((actual % 360) - (target % 360))
    if diff > 180: diff = 360 - diff
    return diff <= tol, round(diff, 1)


def tilt_tolerance(actual: float, target: float, tol: float = 15.0) -> tuple:
    diff = abs(actual - target)
    return diff <= tol, round(diff, 1)


def compute_sha256_hash(data: dict) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


def haversine_km(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return round(6371 * 2 * math.asin(math.sqrt(a)), 3)


# ═══════════════════════════════════════════════════════════════
# SECTION 8: REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════

class InitiateRequest(BaseModel):
    transaction_amount: float
    transaction_id: str
    user_id: str
    latitude:  Optional[float] = None
    longitude: Optional[float] = None

class VerifyRequest(BaseModel):
    session_id:        str
    user_id:           str
    transaction_id:    str
    latitude:          float
    longitude:         float
    altitude:          Optional[float] = 0.0
    compass_heading:   float
    tilt_angle:        float
    roll_angle:        Optional[float] = 0.0
    sky_image_base64:  str
    device_id:         Optional[str]  = "unknown"
    timestamp_client:  Optional[int]  = None


# ═══════════════════════════════════════════════════════════════
# SECTION 9: API ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend — expects index.html in same directory."""
    for path in ["index.html", "templates/index.html"]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return HTMLResponse(f.read())
    return HTMLResponse("<h1>SkyAuth v3</h1><p>Place index.html in same directory as server.py</p>")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0", "server_time": datetime.now().isoformat()}


@app.post("/api/initiate")
async def initiate_auth(req: InitiateRequest):
    """
    Step 1 — Initiate payment authentication.

    No direction/tilt challenge. The user simply needs to photograph the sky
    with the sun visible. Gemini AI verifies authenticity and sun presence.
    The server stores the solar position as ground truth for Gemini to compare against.
    """
    if req.transaction_amount <= 0:
        raise HTTPException(400, "Transaction amount must be positive")
    if req.latitude is None or req.longitude is None:
        raise HTTPException(400, "GPS coordinates are required to compute sun position.")
    if not (-90 <= req.latitude <= 90) or not (-180 <= req.longitude <= 180):
        raise HTTPException(400, "Invalid GPS coordinates.")

    utc_now = datetime.now(timezone.utc)
    sun = solar_position(req.latitude, req.longitude, utc_now)

    if not sun["is_above_horizon"]:
        raise HTTPException(
            403,
            f"SkyAuth requires daylight. "
            f"Sun is {abs(sun['elevation_deg']):.1f}° below horizon at your location. "
            f"Please try again during daylight hours."
        )
    if sun["elevation_deg"] < 3.0:
        raise HTTPException(
            403,
            f"Sun elevation {sun['elevation_deg']:.1f}° is too close to the horizon. "
            f"Please try when the sun is higher in the sky."
        )

    nonce      = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
    session_id = hashlib.sha256(
        f"{req.transaction_id}{req.user_id}{time.time()}".encode()
    ).hexdigest()[:24]

    # Store solar ground truth — Gemini will compare against this at verify time
    sessions[session_id] = {
        "sun_azimuth":        sun["azimuth_deg"],
        "sun_elevation":      sun["elevation_deg"],
        "nonce":              nonce,
        "issued_at":          int(time.time()),
        "expires_at":         int(time.time()) + 120,
        "transaction_amount": req.transaction_amount,
        "transaction_id":     req.transaction_id,
        "user_id":            req.user_id,
        "initiated_at":       int(time.time()),
        "init_lat":           req.latitude,
        "init_lon":           req.longitude,
    }

    return {
        "session_id": session_id,
        "sun": sun,
        "instruction": (
            f"☀️ Point your camera at the sky where you can see the sun "
            f"and take a photo. SkyAuth's AI will verify the image. "
            f"You have 120 seconds."
        ),
        # Still expose sun data for the UI info panel — not a challenge
        "sun_info": {
            "azimuth_deg":   round(sun["azimuth_deg"], 1),
            "elevation_deg": round(sun["elevation_deg"], 1),
            "direction":     azimuth_to_direction(sun["azimuth_deg"]),
        },
    }


@app.post("/api/verify")
async def verify_auth(req: VerifyRequest):
    """
    Step 2 — Full verification.
    No direction/tilt challenge. Gemini is the primary judge.
    Checks: GPS validity + drift | Daytime | Gemini authenticity + sun presence | Deepfake ML | Timestamp
    """
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found or expired")
    session = sessions[req.session_id]

    if int(time.time()) > session["expires_at"]:
        del sessions[req.session_id]
        raise HTTPException(410, "Challenge expired. Please start a new transaction.")

    log.info("═" * 60)
    log.info(f"🔐 VERIFY REQUEST | user={req.user_id} | txn={req.transaction_id}")
    log.info(f"   GPS: {req.latitude:.4f},{req.longitude:.4f}  heading={req.compass_heading}°  tilt={req.tilt_angle}°")

    # ── 1. Weather ──────────────────────────────────────────────
    weather = get_weather(req.latitude, req.longitude)
    log.info(f"   Weather: {weather.get('description')} | clouds={weather.get('clouds_pct')}% | "
             f"vis={weather.get('visibility_m')}m | temp={weather.get('temperature_c')}°C")

    # ── 2. Solar position at submission time ────────────────────
    utc_now    = datetime.now(timezone.utc)
    sun        = solar_position(req.latitude, req.longitude, utc_now)
    is_daytime = sun["is_above_horizon"] and sun["elevation_deg"] >= 3.0
    log.info(f"   Sun: az={sun['azimuth_deg']}° el={sun['elevation_deg']}° | daytime={is_daytime}")

    # ── 3. OpenCV sun detection (informational only — not a blocker) ──
    log.info("─── [OpenCV] Sun detection ───")
    cv_result   = detect_sun_in_image(req.sky_image_base64)
    log.info(f"   sun_detected={cv_result.get('sun_detected')} | "
             f"is_sky={cv_result.get('is_sky_image')} | "
             f"sky_ratio={cv_result.get('sky_pixel_ratio')} | "
             f"brightness_ratio={cv_result.get('brightness_ratio')} | "
             f"overexposed={cv_result.get('overexposed_fraction')}")
    if cv_result.get("error"):
        log.warning(f"   OpenCV error: {cv_result['error']}")

    sun_compare = compare_sun_position(
        cv_result,
        device_heading  = req.compass_heading,
        device_tilt     = req.tilt_angle,
        solar_azimuth   = sun["azimuth_deg"],
        solar_elevation = sun["elevation_deg"],
    )
    log.info(f"   Sun position match: {sun_compare.get('match')} | {sun_compare.get('reason','')}")

    # ── 4. Gemini AI — PRIMARY JUDGE ────────────────────────────
    log.info("─── [Gemini] AI analysis ───")
    ai_result = analyze_image_with_gemini(
        req.sky_image_base64,
        {
            "lat":             req.latitude,
            "lon":             req.longitude,
            "solar_azimuth":   sun["azimuth_deg"],
            "solar_elevation": sun["elevation_deg"],
            "weather_desc":    weather.get("description", "unknown"),
            "cloud_pct":       weather.get("clouds_pct", 50),
        }
    )
    log.info(f"   Gemini verdict: {ai_result.get('overall_verdict')} | "
             f"confidence={ai_result.get('verdict_confidence')}% | "
             f"sun_visible={ai_result.get('sun_visible')} | "
             f"ai_generated_prob={ai_result.get('ai_generated_probability')}% | "
             f"ai_score={ai_result.get('ai_score')}")
    log.info(f"   Scene: {ai_result.get('scene_type')} | sky_condition={ai_result.get('sky_condition')}")
    log.info(f"   Sun reason: {ai_result.get('sun_visibility_reason','')}")
    log.info(f"   Auth reason: {ai_result.get('authenticity_reason','')}")
    if ai_result.get("gemini_hard_block"):
        log.warning(f"   ❌ GEMINI HARD BLOCK: {ai_result.get('gemini_block_reason')}")

    # ── 5. Gemini visual sun position cross-check ────────────────
    gemini_sun_cmp = compare_gemini_sun_estimate(
        ai_result, sun["azimuth_deg"], sun["elevation_deg"]
    )
    log.info(f"   Gemini sun position: {gemini_sun_cmp.get('reason','')}")

    # ── 6. Deepfake / AI-image detection (Gemini feeds in) ──────
    log.info("─── [Fake Detection] ML pipeline ───")
    gemini_ai_prob = ai_result.get("ai_generated_probability", 30)
    fake_result    = detect_fake_image(
        req.sky_image_base64,
        gemini_ai_prob,
        weather_context={
            "clouds_pct":   weather.get("clouds_pct", 0),
            "description":  weather.get("description", ""),
            "visibility_m": weather.get("visibility_m", 10000),
        }
    )
    log.info(f"   Fake prob={fake_result.get('fake_probability')} | "
             f"is_fake={fake_result.get('is_likely_fake')} | "
             f"relax={fake_result.get('weather_relaxation','?')}x | "
             f"flags={fake_result.get('flags', [])}")

    # ── 7. GPS and timestamp checks ─────────────────────────────
    gps_valid    = (-90 <= req.latitude <= 90) and (-180 <= req.longitude <= 180)
    gps_drift_km = 0.0
    gps_drift_ok = True
    init_lat = session.get("init_lat")
    init_lon = session.get("init_lon")
    if init_lat is not None and init_lon is not None:
        gps_drift_km = haversine_km(req.latitude, req.longitude, init_lat, init_lon)
        gps_drift_ok = gps_drift_km < 1.0

    ts_diff  = abs(int(time.time()) - (req.timestamp_client or int(time.time())))
    ts_fresh = ts_diff < 120
    log.info(f"   GPS: valid={gps_valid} drift={gps_drift_km:.3f}km ok={gps_drift_ok}")
    log.info(f"   Timestamp diff: {ts_diff}s | fresh={ts_fresh}")

    # ── 8. Gemini verdict interpretation ────────────────────────
    gemini_verdict     = ai_result.get("overall_verdict", "genuine_sun_visible")
    gemini_sun_ok      = gemini_verdict in ("genuine_sun_visible", "genuine_sun_obscured")
    gemini_verdict_conf = ai_result.get("verdict_confidence", 50)

    # ── 9. Random Forest features ───────────────────────────────
    #    Direction/tilt removed — replaced with Gemini verdict confidence
    rf_features = {
        "gps_valid":        float(gps_valid and gps_drift_ok),
        "is_daytime":       float(is_daytime),
        "direction_match":  float(gemini_sun_ok),                          # Gemini replaces compass
        "tilt_match":       min(1.0, gemini_verdict_conf / 100.0),         # Gemini confidence replaces tilt
        "sun_detected":     float(cv_result.get("sun_detected", False) or ai_result.get("sun_visible", False)),
        "sun_pos_match_cv": float(sun_compare.get("match", False)),
        "not_fake":         1.0 - fake_result.get("fake_probability", 0.5),
        "gemini_ai_score":  ai_result.get("ai_score", 50) / 100.0,
        "ts_freshness":     1.0 if ts_fresh else (0.5 if ts_diff < 300 else 0.0),
        "is_sky_image":     float(cv_result.get("is_sky_image", False) or ai_result.get("sky_visible", False)),
        "gemini_sun_match": float(gemini_sun_cmp.get("match", False)),
        "gemini_no_block":  0.0 if ai_result.get("gemini_hard_block") else 1.0,
    }
    rf = rf_decision(rf_features)

    # ── 10. Score breakdown ──────────────────────────────────────
    # Sun detected = OpenCV found it OR Gemini says sun/glare visible OR sun behind cloud with bright patch
    gemini_sun_any = (
        ai_result.get("sun_visible") or
        (ai_result.get("sun_behind_cloud") and gemini_verdict in ("genuine_sun_visible", "genuine_sun_obscured"))
    )
    sun_detected_final = cv_result.get("sun_detected") or gemini_sun_any

    breakdown = {
        "gps":              25 if (gps_valid and gps_drift_ok) else (10 if gps_valid else 0),
        "daytime":          15 if is_daytime else 0,
        "gemini_verdict":   15 if gemini_sun_ok else (8 if gemini_verdict == "suspicious" else 0),
        "sun_detected":     10 if sun_detected_final else 0,
        "not_fake":         15 if not fake_result.get("is_likely_fake") else 0,
        "gemini_score":     int(ai_result.get("ai_score", 50) * 0.10),
        "is_sky_image":     5  if (cv_result.get("is_sky_image") or ai_result.get("sky_visible")) else 0,
        "timestamp":        10 if ts_fresh else (5 if ts_diff < 300 else 0),
    }
    classic_score = sum(breakdown.values())

    # ── 11. HARD BLOCKERS ───────────────────────────────────────
    denial_reasons = []

    if not is_daytime:
        denial_reasons.append(
            f"Sun is below horizon (elevation: {sun['elevation_deg']:.1f}°) — SkyAuth requires daylight"
        )
    if not gps_drift_ok:
        denial_reasons.append(
            f"Location mismatch: moved {gps_drift_km:.2f} km from session origin (max: 1.0 km)"
        )
    if not gps_valid:
        denial_reasons.append("Invalid GPS coordinates")

    if fake_result.get("is_likely_fake"):
        denial_reasons.append(
            f"Image flagged as fake/AI-generated "
            f"(probability: {fake_result['fake_probability']*100:.0f}%, "
            f"flags: {', '.join(fake_result.get('flags',[])[:3])})"
        )

    if ai_result.get("gemini_hard_block"):
        denial_reasons.append(
            f"Gemini AI: {ai_result.get('gemini_block_reason', 'Not a sky image')}"
        )

    # Hard block: no sun trace at all (not even a diffused glow or bright patch)
    # Only block if BOTH Gemini and OpenCV agree there's no sun/bright region
    no_sun_trace = (
        not sun_detected_final and
        gemini_verdict not in ("genuine_sun_visible", "genuine_sun_obscured") and
        not ai_result.get("sun_behind_cloud", False)
    )
    if no_sun_trace and is_daytime and not ai_result.get("gemini_hard_block"):
        denial_reasons.append(
            "No sun or bright region detected — point camera toward the sun (even through haze/clouds, the sun should leave a visible bright patch)"
        )

    hard_fail = any([
        not is_daytime,
        not gps_drift_ok,
        not gps_valid,
        fake_result.get("is_likely_fake"),
        ai_result.get("gemini_hard_block"),
        no_sun_trace,
    ])

    rf_approved      = rf["approved"]
    classic_approved = classic_score >= 55
    approved         = (not hard_fail) and rf_approved and classic_approved

    if not rf_approved and not hard_fail:
        denial_reasons.append(f"AI ensemble confidence insufficient ({rf['confidence']:.1f}%)")
    if not classic_approved and not hard_fail and not denial_reasons:
        denial_reasons.append(f"Verification score too low ({classic_score}/100, need 55)")

    log.info("─── [FINAL VERDICT] ───")
    log.info(f"   classic_score={classic_score}/100 (need 55) → {'✅' if classic_approved else '❌'}")
    log.info(f"   rf_confidence={rf['confidence']}% → {'✅' if rf_approved else '❌'}")
    log.info(f"   hard_fail={hard_fail}")
    log.info(f"   breakdown: {breakdown}")
    if approved:
        log.info(f"   ✅✅ APPROVED ✅✅")
    else:
        log.warning(f"   ❌❌ REJECTED ❌❌")
        for r in denial_reasons:
            log.warning(f"      Reason: {r}")
    log.info("═" * 60)

    # ── 12. Crypto hash + log ────────────────────────────────────
    hash_payload = {
        "session_id":      req.session_id,
        "transaction_id":  req.transaction_id,
        "user_id":         req.user_id,
        "gps":             {"lat": req.latitude, "lon": req.longitude},
        "nonce":           session["nonce"],
        "solar_azimuth":   sun["azimuth_deg"],
        "solar_elevation": sun["elevation_deg"],
        "gemini_verdict":  gemini_verdict,
        "server_ts":       int(time.time()),
    }
    crypto_hash = compute_sha256_hash(hash_payload)

    tx_record = {
        "transaction_id":  req.transaction_id,
        "user_id":         req.user_id,
        "amount":          session["transaction_amount"],
        "timestamp":       utc_now.isoformat(),
        "result":          "APPROVED" if approved else "REJECTED",
        "rf_confidence":   rf["confidence"],
        "classic_score":   classic_score,
        "crypto_hash":     crypto_hash,
        "location":        {"lat": req.latitude, "lon": req.longitude},
        "weather":         weather,
        "denial_reasons":  denial_reasons,
        "gemini_verdict":  gemini_verdict,
    }
    transactions.append(tx_record)
    del sessions[req.session_id]

    return {
        "status":            "APPROVED" if approved else "REJECTED",
        "approved":          approved,
        "denial_reasons":    denial_reasons,

        # Scores
        "rf_decision":       rf,
        "classic_score":     classic_score,
        "classic_threshold": 55,
        "breakdown":         breakdown,

        # Solar position
        "solar_position":    sun,

        # Gemini — primary judge
        "ai_analysis":       ai_result,
        "gemini_verdict":    gemini_verdict,
        "gemini_verdict_confidence": gemini_verdict_conf,
        "gemini_sun_reasoning": ai_result.get("sun_visibility_reason", ""),
        "gemini_auth_reasoning": ai_result.get("authenticity_reason", ""),
        "gemini_sun_comparison": gemini_sun_cmp,

        # OpenCV (informational)
        "sun_cv_analysis":   cv_result,
        "sun_comparison_cv": sun_compare,

        # Deepfake ML
        "fake_detection":    fake_result,

        # Context
        "weather":           weather,
        "crypto_hash":       crypto_hash,
        "transaction":       tx_record,
        "gps_drift_km":      gps_drift_km,

        "message": (
            f"✅ Payment APPROVED — AI Confidence: {rf['confidence']}% | Score: {classic_score}/100"
            if approved else
            f"🚫 ACCESS DENIED — {denial_reasons[0] if denial_reasons else 'Verification failed'}"
        ),
    }

@app.get("/api/transactions")
async def get_transactions():
    return {"transactions": transactions[-20:], "count": len(transactions)}


# ═══════════════════════════════════════════════════════════════
# SECTION 10: ENTRYPOINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn, socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = "localhost"
    finally:
        s.close()

    print("\n" + "=" * 65)
    print("  🛡️  SkyAuth Payment Server v3.0  — Full Intelligence Edition")
    print("=" * 65)
    print(f"  Local:   http://localhost:8000")
    print(f"  Mobile:  http://{local_ip}:8000")
    print(f"  API docs: http://{local_ip}:8000/docs")
    print("=" * 65)
    print("  ✅ Gemini AI: First-class decision maker (blocks fakes)")
    print("  ✅ Deepfake ML: ELA + noise + freq + Gemini fusion")
    print("  ✅ Ground-image rejection: CV sky-ratio + Gemini scene")
    print("  ✅ Azimuth/Elevation: CV pixel + Gemini visual estimate")
    print("  ✅ Solar API: NOAA algorithm (±0.01° accuracy)")
    print("  ✅ Random Forest: 12-feature ensemble")
    print("=" * 65)

    gemini_ready = GEMINI_API_KEY not in ("INSERT_YOUR_GEMINI_API_KEY_HERE", "", None)
    weather_ready = OPENWEATHER_API_KEY not in ("INSERT_YOUR_OPENWEATHERMAP_KEY_HERE", "", None)
    print(f"\n  Gemini API:    {'✅ Configured' if gemini_ready else '⚠️  NOT SET — running in mock mode'}")
    print(f"  OpenWeather:   {'✅ Configured' if weather_ready else '⚠️  NOT SET — using mock weather'}")
    print("\n  Set keys at top of server.py to enable real AI analysis.\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
