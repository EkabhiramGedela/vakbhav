# VaakBhav — Hinglish Sentiment Analyzer

A full-stack web app for dual-language (Hindi + English / Hinglish) sentiment analysis.
Features a **3D animated landing page** + a fully-featured analyzer UI with batch processing.

---

## Quick Local Start

```bash
# 1. Clone / download the repo
git clone https://github.com/YOUR_USERNAME/vaakbhav.git
cd vaakbhav

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (creates models/ folder — takes 1–3 min)
python train_model.py

# 4. Run
python app.py
```

Open http://localhost:5000

---

## Folder Structure

```
vaakbhav/
├── app.py                  ← Flask app
├── train_model.py          ← ML training script
├── build.sh                ← Cloud build script (installs NLTK + trains)
├── requirements.txt        ← All Python deps
├── Procfile                ← Heroku / Railway / Render process file
├── railway.json            ← Railway-specific config
├── render.yaml             ← Render-specific config
├── runtime.txt             ← Python version
├── nixpacks.toml           ← Railway Nixpacks build config
├── .replit                 ← Replit run config
├── replit.nix              ← Replit Nix deps
├── vercel.json             ← Vercel config (VADER-fallback only)
├── output__1_.csv          ← Training data (keep in repo root)
├── models/                 ← Auto-created after training
│   ├── sentiment_model.pkl
│   ├── char_vectorizer.pkl
│   └── label_encoder.pkl
└── templates/              ← Flask template folder (do NOT rename)
    ├── landing.html        ← served at GET /
    └── index.html          ← served at GET /app
```

---

## ☁️ Deployment Guide

### ✅ Option 1 — Railway (Recommended, easiest)

1. Push repo to GitHub
2. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub**
3. Select your repo
4. Railway auto-detects `railway.json` — no manual config needed
5. In **Settings → Environment**, add:
   ```
   FLASK_ENV=production
   ```
6. Railway will:
   - Run `pip install -r requirements.txt`
   - Run `bash build.sh` (downloads NLTK data + trains model)
   - Start gunicorn automatically
7. Your site is live at the Railway-provided URL ✅

> **Build time:** ~3–5 minutes (model training). Free tier has 500 hrs/month.

---

### ✅ Option 2 — Render

1. Push repo to GitHub
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your GitHub repo
4. Render auto-reads `render.yaml`. Confirm these settings:
   - **Build Command:** `pip install -r requirements.txt && bash build.sh`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 2 --preload`
   - **Runtime:** Python 3
5. Click **Create Web Service**

> Free tier spins down after 15 min inactivity — first request after sleep takes ~30s.

---

### ✅ Option 3 — Heroku

```bash
# Install Heroku CLI, then:
heroku create vaakbhav-app
heroku buildpacks:set heroku/python
git push heroku main

# After deploy, run build manually once:
heroku run bash build.sh
heroku restart
```

Or add a `release` phase to `Procfile`:
```
release: bash build.sh
web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 2 --preload
```

> Note: Free dynos were discontinued. Use Eco ($5/mo) or Basic ($7/mo).

---

### ✅ Option 4 — Replit

1. Go to [replit.com](https://replit.com) → **Create Repl → Import from GitHub**
2. Paste your repo URL
3. Replit reads `.replit` automatically
4. Click **Run** — it will build and start the server
5. Use **Deployments → Autoscale** for a persistent public URL

> Free Replit projects sleep after inactivity. Use Always On or paid plan for 24/7.

---

### ✅ Option 5 — PythonAnywhere

1. Log in → **Bash console:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/vaakbhav.git
   cd vaakbhav
   pip3.11 install --user -r requirements.txt
   bash build.sh
   ```
2. Go to **Web** tab → **Add a new web app**
3. Choose **Manual configuration** → Python 3.11
4. Set:
   - **Source code:** `/home/YOUR_USERNAME/vaakbhav`
   - **Working directory:** `/home/YOUR_USERNAME/vaakbhav`
   - **WSGI file:** Edit it to contain:
     ```python
     import sys
     sys.path.insert(0, '/home/YOUR_USERNAME/vaakbhav')
     from app import app as application
     ```
5. Reload the web app ✅

> Free tier: 1 web app, sleeps after inactivity. Paid plan for always-on.

---

### ⚠️ Option 6 — Vercel (Limited — VADER fallback only)

Vercel's 50 MB Lambda limit means the trained `.pkl` model files (~40–60 MB) often won't fit. The app will run using VADER-based fallback sentiment (no ML model).

If you still want Vercel:
1. Push to GitHub
2. Import to [vercel.com](https://vercel.com)
3. Framework: **Other**
4. It uses `vercel.json` automatically

---

## API Endpoints

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `GET` | `/` | — | Landing page |
| `GET` | `/app` | — | Analyzer UI |
| `GET` | `/health` | — | `{"status":"ok","model":true,"nlp":true}` |
| `POST` | `/predict` | `{"text":"..."}` | Single text sentiment |
| `POST` | `/word-batch` | `{"text":"..."}` | Paragraph → per-sentence |
| `POST` | `/word-batch` | `{"texts":["..."]}` | List of texts |
| `POST` | `/word-batch` | multipart `file` | CSV/TXT file upload |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Port to bind (set automatically by hosting platforms) |
| `FLASK_ENV` | `production` | Set to `development` for debug mode locally |

---

## Features

| Tab | What it does |
|-----|-------------|
| **Landing** | 3D animated orbiting letters (Hindi + English) |
| **Analyzer** | Single text prediction — up to 500 words |
| **Batch & Upload** | Paragraph, Text List, or File Upload (up to 1M rows) |
| **Dashboard** | Session history + donut sentiment chart |
| **About** | FAQs, Privacy, Contact, App Info |

---

## Troubleshooting

**App starts but shows "model not loaded"**
→ The `.pkl` files weren't built. Check your build logs for errors from `build.sh`.

**NLTK errors on startup**
→ The app will auto-download NLTK data to `./nltk_data/` on first run. Ensure the process has write access.

**Port already in use locally**
→ `PORT=8080 python app.py`

**Slow first response on Render/Replit free tier**
→ Normal — free instances sleep. First request after sleep takes 20–30s.
