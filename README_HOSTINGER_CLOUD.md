# Quantum Matchmaker — Cloud Hosting (Hostinger DB)

This repo is meant to run the *Streamlit* app on a Python-friendly host (Streamlit Community Cloud, Render, etc.)
while using your **Hostinger MariaDB/MySQL** as the **shared database**.

## Files
- `quantum_matchmaker_app_sql.py` — the multi-user app (reads `DATABASE_URL`).
- `requirements.txt` — Python deps.
- `.streamlit/secrets.toml` — put your DB URL here (Streamlit Cloud reads it).
- `Procfile` — for Render / Heroku-style dynos.

## 1) Create Hostinger database (hPanel)
1. Websites → *Your site* → **Databases → MySQL Databases** → Create DB + User.
2. Note these:
   - DB host (like `sqlxxx.main-hosting.com`), port `3306`
   - DB name (often prefixed, e.g., `u123456789_mydb`)
   - DB username (prefixed similarly)
   - DB password
3. Databases → **Remote MySQL** → allow connections from your app’s IP (or "Any host" if your app’s IP is dynamic).

## 2) Deploy the Streamlit app
### Option A: Streamlit Community Cloud
- Push this repo to GitHub.
- Create a new app from your repo.
- In **App → Settings → Secrets**, paste:
  ```toml
  DATABASE_URL="mysql+pymysql://DB_USER:DB_PASS@DB_HOST:3306/DB_NAME"
  ```
- App will start; share its URL.

### Option B: Render
- Create a Web Service → Build from repo.
- Runtime: Python 3.x.
- Start command:
  ```
  streamlit run quantum_matchmaker_app_sql.py --server.port $PORT --server.address 0.0.0.0
  ```
- Add env var `DATABASE_URL` with the same connection string.

## 3) Point your Hostinger site to the app
On your Hostinger website, add a simple link button to the Streamlit URL, or proxy via Cloudflare/NGINX if you want a custom domain URL.
