@echo off
setlocal enabledelayedexpansion

echo ===========================================
echo ArtiQuest Development Environment Setup & Run
echo ===========================================
echo.

:: Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

:: Check if Node.js is installed
where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js is not installed or not in PATH. Please install Node.js.
    pause
    exit /b 1
)

:: Check if npm is installed
where npm >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] npm is not installed or not in PATH. Please install npm.
    pause
    exit /b 1
)

:: Create .env file if it doesn't exist
if not exist ".env" (
    echo [INFO] Creating .env file...
    (
        echo # Application Settings
        echo DEBUG=True
        echo ENVIRONMENT=development
        echo SECRET_KEY=your-secret-key-here
        echo 
        echo # Database Settings
        echo DATABASE_URL=sqlite:///artifacts.db
        echo 
        echo # Email Settings
        echo SMTP_SERVER=smtp.example.com
        echo SMTP_PORT=587
        echo SMTP_USERNAME=your-email@example.com
        echo SMTP_PASSWORD=your-email-password
        echo DEFAULT_FROM_EMAIL=your-email@example.com
    ) > .env
    echo [SUCCESS] .env file created with default values.
) else (
    echo [INFO] .env file already exists. Using existing configuration.
)

:: Initialize database
echo.
echo [INFO] Initializing database...
cd MainApp
python -c "from database import Base, engine; Base.metadata.create_all(bind=engine)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to initialize database.
    pause
    exit /b 1
)
cd ..
echo [SUCCESS] Database initialized successfully.

:: Install Python dependencies
echo.
echo [INFO] Installing Python dependencies...
pip install -r MainApp\requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install Python dependencies.
    pause
    exit /b 1
)
echo [SUCCESS] Python dependencies installed.

:: Install Node.js dependencies
echo.
echo [INFO] Installing Node.js dependencies...
cd frontend
call npm install
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install Node.js dependencies.
    cd ..
    pause
    exit /b 1
)
cd ..
echo [SUCCESS] Node.js dependencies installed.

:: Start the application
echo.
echo ===========================================
echo Starting ArtiQuest Application...
echo ===========================================
echo.

:: Start backend server
start "ArtiQuest Backend" cmd /k "cd MainApp\backend && python main.py"

timeout /t 3 /nobreak >nul

:: Start frontend server
start "ArtiQuest Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ===========================================
echo Application is starting...
echo ===========================================
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo You can now access the application in your browser.
echo.
pause
