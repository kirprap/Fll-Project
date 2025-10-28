@echo off
setlocal
echo [DEBUG] Starting Docker-based setup...

set "ERROR_OCCURRED=0"

REM -- Error handler: call with a message to pause and record error
:handle_error
if "%~1" neq "" echo.
if "%~1" neq "" echo [ERROR] %~1
set "ERROR_OCCURRED=1"
echo.
echo Press any key to inspect the console. After inspection, press Enter to continue running the script.
pause >nul
goto :eof

REM -- Check for Docker availability
docker --version >nul 2>&1
if errorlevel 1 (
    call :handle_error "Docker is not available on PATH. Install Docker Desktop and ensure 'docker' is on PATH."
    goto :end
)

REM -- Prefer docker compose if a compose file exists
if exist docker-compose.yml (
    echo [DEBUG] Found docker-compose.yml — using Docker Compose workflow.

    rem Try to pull images (best-effort)
    docker compose pull
    if errorlevel 1 (
        call :handle_error "'docker compose pull' failed or no remote images. Continuing with build."
    )

    rem Build services
    echo [DEBUG] Building services with Docker Compose...
    docker compose build --no-cache
    if errorlevel 1 (
        call :handle_error "Docker Compose build failed."
        goto :end
    )

    rem Start services (attached so user sees logs). Use Ctrl+C to stop.
    echo [DEBUG] Starting services (attached). Press Ctrl+C to stop and then run 'docker compose down' if needed.
    docker compose up --force-recreate
    if errorlevel 1 (
        call :handle_error "Docker Compose 'up' ended with an error."
        goto :end
    )

    echo [DEBUG] Docker Compose services stopped.
    goto :end
)

REM -- No compose file: build image from Dockerfile and run container
echo [DEBUG] No docker-compose.yml found — building Docker image 'fll-project:local'...
docker build -t fll-project:local .
if errorlevel 1 (
    call :handle_error "Docker build failed."
    goto :end
)

REM -- Run container and map port 8501 (Streamlit). Mount current directory for live development.
echo [DEBUG] Running container 'fll-project:local' (port 8501 -> 8501). Press Ctrl+C to stop.
docker run --rm -it -p 8501:8501 -v "%CD%":/app --env-file .env fll-project:local
if errorlevel 1 (
    call :handle_error "Docker container exited with an error."
    goto :end
)

echo [DEBUG] Docker container stopped normally.

:end
echo.
if "%ERROR_OCCURRED%"=="1" (
    echo [DEBUG] One or more errors occurred during the run. The script will remain open so you can inspect logs.
) else (
    echo [DEBUG] Run completed successfully.
)
echo.
echo Press any key to close this window.
pause >nul
endlocal