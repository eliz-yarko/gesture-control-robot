# Deployment Notes

## Recommended Diploma Setup

Use two deployment modes:

1. Static demo frontend for a public link.
2. Local live backend for camera, MediaPipe, and command generation.

This split is intentional: cloud hosting cannot access the laptop/Raspberry Pi camera directly,
while the static page is enough to show the operator interface and interaction model.

## Static Demo

Export the frontend:

```powershell
python scripts/export_demo_frontend.py --output docs/demo
```

Then deploy `docs/demo/` with one of these free options:

- GitHub Pages from the repository `docs/` folder.
- Netlify free site from the `docs/demo/` directory.
- Vercel Hobby project with `docs/demo/` as static output.

The static page automatically switches to demo data when `/api/status` is unavailable.

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

- GitHub Pages for the static demo page.
- Local web UI for the actual camera demo.
- Cloudflare Tunnel only if a public live URL is required for a short presentation window.
