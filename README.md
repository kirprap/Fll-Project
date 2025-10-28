# Archaeological Artifact Identifier (Fll-Project)

This project is a Streamlit web app that uses Hugging Face vision models to analyze images of archaeological artifacts. It supports local development using a Python virtual environment and deployment using Docker / docker-compose. The app includes a small PostgreSQL-backed archive for saved artifacts.

## Features
- Upload photos and get AI-powered identifications
- Batch processing of multiple images
- Archive of identified artifacts (PostgreSQL)
- Visual similarity search using CLIP-style embeddings
- Expert verification and profile editing UI

---

## Prerequisites

- Python 3.11
- Docker & Docker Compose (if you want to run with containers)
- Git (optional)
- A Hugging Face API token if you plan to use authenticated HF Inference (recommended for reliability and rate limits)

Optional environment variables used by the project:

- `HUGGINGFACE_TOKEN` — (recommended) Hugging Face Inference token. Used by `ai_analyzer.py` to create an authenticated `InferenceClient`.
- `GEMINI_API_KEY` — present in `docker-compose.yml` as an environment placeholder; not required by the app unless you integrate Google Gemini features separately.

The `docker-compose.yml` config will create a local Postgres database and expose the web UI on port `8501` by default.

---

## Quickstart — Local (Python virtualenv)

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv env
& .\env\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

3. (Optional) Set your Hugging Face token in the current PowerShell session:

```powershell
$env:HUGGINGFACE_TOKEN = "hf_...your_token_here..."
```

4. Run the Streamlit app:

```powershell
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

5. Open your browser to `http://localhost:8501`.

Notes:
- If you run the app locally and also use Docker, ensure only one instance is binding port `8501` at a time.

---

## Quickstart — Docker Compose

This repository contains a `Dockerfile` and `docker-compose.yml` to build and run the web app and a Postgres database.

1. (Optional) Create an `.env` file in the repo root to provide environment variables used by compose (e.g. `GEMINI_API_KEY` or `HUGGINGFACE_TOKEN`):

```
HUGGINGFACE_TOKEN=hf_...your_token_here...
GEMINI_API_KEY=
```

2. Build and start the services (PowerShell):

```powershell
docker-compose build --no-cache
docker-compose up -d
```

3. Check logs:

```powershell
docker-compose logs -f
```

4. Stop services:

```powershell
docker-compose down
```

The web UI will be available at `http://localhost:8501` unless you changed the port mapping.

---

## Troubleshooting

- Port 8501 already in use

  If you get an error like `Bind for 0.0.0.0:8501 failed: port is already allocated`, another process or container is using port 8501.

  Find the process using PowerShell:

  ```powershell
  netstat -aon | Select-String ":8501"
  ```

  Map PID to a process name:

  ```powershell
  tasklist /FI "PID eq <PID>"
  ```

  If a Docker container is holding the port, list containers and stop the conflicting one:

  ```powershell
  docker ps
  docker stop <container_id>
  docker rm <container_id>
  ```

  You can also change the host port in `docker-compose.yml` (for example `8510:8501`) to avoid conflicts.

- Syntax errors on startup

  If the startup logs (see `docker-compose logs web`) show a Python syntax error referencing `ai_analyzer.py`, ensure the file in the running container matches the repository copy. Common causes:

  - A stale container image running with an older copy of the code. Solution: `docker-compose down` then `docker-compose build --no-cache` and `docker-compose up -d`.
  - A local process (non-container) running Streamlit on the same port. Stop it (Ctrl+C) or change the port.

- Hugging Face Inference errors (rate limits / auth)

  If you see authentication or rate-limit errors from Hugging Face, set `HUGGINGFACE_TOKEN` in your environment (PowerShell) before starting the app, or add it to `.env` for docker-compose.

  Example (PowerShell):

  ```powershell
  $env:HUGGINGFACE_TOKEN = "hf_...your_token_here..."
  docker-compose up -d --build
  ```

---

## Development tips

- The `docker-compose.yml` mounts the repository directory into the container (`.:/app`), which means code changes on the host are reflected inside the container immediately. For changes to `requirements.txt` or installed system packages, rebuild the image.
- Add a quick pre-build syntax check to avoid shipping syntax errors. Example snippet you can add to CI or the Dockerfile during build:

```dockerfile
RUN python -m py_compile $(python -c "import pathlib,sys; print(' '.join([str(p) for p in pathlib.Path('/app').rglob('*.py')]))") || true
```

---

## Verifying the installation

- Quick import test (inside venv):

```powershell
python -c "from ai_analyzer import analyze_artifact_image; print('ai_analyzer import OK')"
```

- Check Streamlit runs and opens the UI at `http://localhost:8501`.

---

## Project layout (key files)

- `app.py` — Streamlit UI and main entrypoint
- `ai_analyzer.py` — AI analysis helpers (Hugging Face Inference client usage, CLIP embedding helpers)
- `artifact_database.py`, `database.py` — persistence / DB helpers
- `Dockerfile`, `docker-compose.yml` — containerization
- `requirements.txt` — Python dependencies

---

If you'd like, I can add a small PowerShell helper script (`dev.ps1`) to streamline common tasks (create venv, install deps, run app, or safely restart docker-compose and free port 8501). Would you like me to add that next?

---

License: Your choice — this repository currently has no license file.
