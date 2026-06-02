# Deployment Notes

## Recommended Diploma Setup

Use three deployment modes:

1. Static demo frontend for a public link.
2. Browser-camera frontend with a Python backend.
3. Local live backend for server-side camera, MediaPipe, and command generation.

The browser-camera mode is the preferred deployed architecture: the website asks the user for
camera permission and sends compressed JPEG frames to the Python backend. The backend runs
MediaPipe, gesture classification, debouncing, and command mapping.

For deployed browser-camera mode, use HTTPS for both the frontend and backend. Browsers allow
camera access on `localhost` during local testing, but public pages need a secure context, and an
HTTPS frontend will block requests to a plain HTTP backend as mixed content.

## Static Demo

Export the frontend:

```powershell
python scripts/export_demo_frontend.py --output docs/demo
```

Then deploy `docs/demo/` with one of these free options:

- GitHub Pages from the repository `docs/` folder.
- Netlify free site from the `docs/demo/` directory.
- Vercel Hobby project with `docs/demo/` as static output.

The static page automatically switches to demo data when `/api/status` is unavailable. For a
deployed backend, open the static page with:

```text
https://your-static-site.example/?api=https://your-backend.example
```

The same URL can also be typed into the `Backend URL` field in the UI.

## Browser Camera + Backend

Run locally:

```powershell
python -m src.web_ui `
  --input-mode browser-camera `
  --host 127.0.0.1 `
  --port 8000 `
  --static-model models/static_gesture_classifier.joblib `
  --dynamic-model models/dynamic_gesture_classifier.joblib `
  --cors-origin "*"
```

Open:

```text
http://127.0.0.1:8000
```

Press `Camera` and allow browser camera access.

Run with deployment environment variables:

```powershell
$env:PORT="8000"
$env:CORS_ORIGIN="*"
python scripts/run_browser_backend.py
```

The backend exposes:

- `GET /api/status` for dashboard state.
- `GET /api/commands` for the command log.
- `POST /api/frame` for browser-uploaded JPEG frames.
- `GET /stream.mjpg` for the latest annotated backend frame.

## Container Backend

Build and run locally:

```powershell
docker build -t gesture-control-backend .
docker run --rm -p 8000:8000 `
  -e PORT=8000 `
  -e CORS_ORIGIN="*" `
  gesture-control-backend
```

For a one-month diploma demo, use Render Free for the hosted backend if cold starts after idle
traffic are acceptable. Railway is convenient for Docker deployments, but use it only when a small
monthly cost is acceptable.

- Deploy from the GitHub repository.
- Use the Dockerfile in the repository root.
- Set `PORT` according to the platform if it is not injected automatically.
- Set `CORS_ORIGIN` to the frontend origin, or `*` for a temporary demo.
- Keep the service in browser-camera mode through `scripts/run_browser_backend.py`.

## Live Local Demo

Run the real backend locally:

```powershell
python -m src.web_ui `
  --camera 0 `
  --static-model models/static_gesture_classifier.joblib `
  --dynamic-model models/dynamic_gesture_classifier.joblib `
  --host 127.0.0.1 `
  --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Public Live Link Without Cloud Backend

For a temporary defense demo, expose the local backend through a tunnel:

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

This keeps MediaPipe and the camera on the local machine, while giving reviewers a public URL.
Do not use this mode for sensitive data or long-running unattended access.

## Practical Choice

For the diploma, use:

- Netlify/Vercel for the static frontend.
- Render Free or local Docker for the browser-camera backend.
- Railway only if a small monthly backend cost is acceptable.
- Local server-camera mode only for the offline Python demo.
- Cloudflare Tunnel only if a short public link to the local backend is enough.
