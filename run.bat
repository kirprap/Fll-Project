@echo off
echo [DEBUG] Starting setup process...

echo [DEBUG] Setting up virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

echo [DEBUG] Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

echo [DEBUG] Installing PyTorch...
pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch
    pause
    exit /b 1
)

echo [DEBUG] Installing other requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements
    pause
    exit /b 1
)

echo [DEBUG] Starting Streamlit application...
echo [INFO] Press Ctrl+C in this window to stop the application
echo [INFO] Keep this window open for debug messages

streamlit run app.py
if errorlevel 1 (
    echo [ERROR] Failed to start Streamlit application
    pause
    exit /b 1
)

echo [DEBUG] Application stopped
pause