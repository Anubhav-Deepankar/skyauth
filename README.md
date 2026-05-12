# SkyAuth

Payment-flow authentication server that uses a live sky photo, device sensors, and a solar-position check to verify the user is physically outdoors at the claimed location. Gemini AI scores image authenticity (real photo vs. screenshot / AI-generated / indoor). OpenWeather provides cloud context.

## Requirements

- Python 3.11+ (tested on 3.13 on Windows)
- A Gemini API key — https://aistudio.google.com/apikey
- An OpenWeather API key — https://home.openweathermap.org/api_keys

Both keys have free tiers. Without them the server still boots but the AI and weather checks run in **mock mode** (clear-sky, generic confidence values).

## Setup

```powershell
git clone https://github.com/SnehalJainn/skyauth.git
cd skyauth
pip install fastapi uvicorn numpy requests pydantic opencv-python Pillow scikit-learn
```

## Configure API keys

The server reads keys from environment variables — they are **not** stored in the repo. See `.env.example` for the variable names.

Set them in your current PowerShell session before running:

```powershell
$env:GEMINI_API_KEY = "your-gemini-key"
$env:OPENWEATHER_API_KEY = "your-openweather-key"
# Optional: override the model
# $env:GEMINI_MODEL = "gemini-2.0-flash"
```

On Windows the console may not render emoji characters in the startup banner. If you see a `UnicodeEncodeError`, force UTF-8 stdio:

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
```

## Run

```powershell
python server.py
```

The server prints something like:

```
Local:   http://localhost:8000
Mobile:  http://<your-LAN-IP>:8000
API docs: http://<your-LAN-IP>:8000/docs
```

Open `http://localhost:8000` in a browser for the frontend, or `/docs` for the interactive Swagger UI.

To test from your phone on the same Wi-Fi, use the `Mobile` URL the server prints. You may need to allow Python through Windows Firewall the first time.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Frontend (`templates/index.html`) |
| GET | `/health` | Liveness check |
| POST | `/api/initiate` | Begin a payment session, get sun-aware challenge |
| POST | `/api/verify` | Submit photo + sensor data for verification |
| GET | `/api/transactions` | List recent transactions |

## Notes on the free tiers

- **OpenWeather:** newly created keys can take up to ~2 hours to activate. Until then you'll see `401 Unauthorized` from `api.openweathermap.org`; the server logs the failure and falls back to a degraded weather record (no crash).
- **Gemini:** if you see `429 RESOURCE_EXHAUSTED` with `limit: 0`, the Google Cloud project tied to your key has no free-tier allocation. Easiest fix: delete the key, recreate it in AI Studio, and choose **"Create API key in new project"**.
