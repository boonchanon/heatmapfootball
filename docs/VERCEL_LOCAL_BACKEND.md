# Vercel frontend + local inference backend

This keeps the costly Python inference on the Windows machine. Vercel hosts
only `frontend/`, and each browser calls the local API through Cloudflare
Tunnel.

## 1. Deploy the frontend

Import this repository in Vercel. The included `vercel.json` publishes the
`frontend` folder. After the first deployment, note its exact production URL,
for example `https://pitchside.vercel.app`.

## 2. Start the backend on the Windows machine

Copy `.env.local.example` to `.env.local` and replace `ALLOWED_ORIGINS` with
the Vercel URL. In PowerShell (the values apply only to this terminal):

```powershell
cd D:\HM\sn-gamestate
$env:ALLOWED_ORIGINS = 'https://your-production.vercel.app'
$env:MAX_JOB_MB = '100'
$env:MAX_ANALYSIS_FRAMES = '7200' # optional; default is 7200 (four minutes at 30 FPS)
.venv\Scripts\python.exe tools\web_app.py
```

Check `http://127.0.0.1:8001/healthz`; it must return `{"status":"ok"}`.

## 3. Publish a stable Cloudflare Tunnel

Install `cloudflared`, then authenticate and create a named tunnel. It needs a
domain managed in Cloudflare for a stable public hostname:

```powershell
cloudflared tunnel login
cloudflared tunnel create video-analysis
cloudflared tunnel route dns video-analysis api.your-domain.com
cloudflared tunnel run --url http://localhost:8001 video-analysis
```

Configure the named tunnel's public hostname in the Cloudflare dashboard as
`api.your-domain.com` with service `http://localhost:8001`. For a persistent
Windows setup, install the connector as a service using the tunnel token shown
in that dashboard. Do not use a temporary `trycloudflare.com` URL for real
users: it changes after restart.

## 4. Use it

Open the Vercel URL and enter `https://api.your-domain.com` into the API URL
field. It is stored in that browser. To preset it for every user, set
`window.PITCHSIDE_API_URL` in `frontend/config.js` to that public URL, then
redeploy Vercel.

## Limits and security

- Cloudflare's free proxy limit is **100 MB per upload**. Set
  `MAX_JOB_MB=100`; it takes precedence over the legacy `MAX_UPLOAD_BYTES`.
  Files above that limit receive HTTP 413. Use shorter/compressed clips or
  implement multipart direct upload before supporting larger files.
- `ALLOWED_ORIGINS` is a comma-separated list of exact origins. Keep the
  production Vercel URL as an exact value. For a preview deployment, add that
  exact preview origin only together with authentication (Cloudflare Access or
  an API token), or use a separate API/tunnel. Never allow `*.vercel.app`.
- CORS is not authentication. Before opening this expensive public upload API
  to general users, add authentication/Cloudflare Access and rate limiting.
- The application processes one job at a time. Keep the PC on, prevent sleep,
  and size `MAX_JOB_RSS_MB` for its actual RAM.
