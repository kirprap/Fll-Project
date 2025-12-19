#!/bin/bash
#@echo off
#setlocal enabledelayedexpansion

echo ===========================================
echo "ArtiQuest Development Environment Setup & Run"
echo ===========================================

# Check if Python is installed
which python3 >null 2>&1
if [ $? -ne 0 ] ;then
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.8 or higher.
    sleep 1
    exit 1
fi

# Check if Node.js is installed
which node >null 2>&1
if [ $? -ne 0 ]; then
    echo [ERROR] Node.js is not installed or not in PATH. Please install Node.js.
    sleep 1
    exit 1
fi

# Check if npm is installed
which npm >null 2>&1
if [ $? -ne 0 ]; then
    echo [ERROR] npm is not installed or not in PATH. Please install npm.
    sleep 1
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f  ".env" ];then
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
else 
    echo [INFO] .env file already exists. Using existing configuration.
fi

# Initialize database

echo [INFO] Initializing database...
cd MainApp
python3 -c "from database import Base, engine; Base.metadata.create_all(bind=engine)" 2>null
if [ $? -ne 0 ];then
    echo [ERROR] Failed to initialize database.
    sleep 1
    exit 1
fi
cd ..
echo [SUCCESS] Database initialized successfully.

# Install Python dependencies
echo [INFO] Installing Python dependencies...
pip3 install -r MainApp/requirements.txt
if [ $? -ne 0 ];then
    echo [ERROR] Failed to install Python dependencies.
    sleep 1
    exit 1
fi
echo [SUCCESS] Python dependencies installed.

# Install Node.js dependencies

echo [INFO] Installing Node.js dependencies...
cd frontend
npm install
if [ $? -ne 0 ];then
    echo [ERROR] Failed to install Node.js dependencies.
    cd ..
    sleep 1
    exit 1
fi
cd ..
echo [SUCCESS] Node.js dependencies installed.

# Start the application

echo ===========================================
echo Starting ArtiQuest Application...
echo ===========================================


# Start backend server
echo "ArtiQuest Backend"  
cd  MainApp/backend
nohup python3  main.py &
cd -

sleep  3 

# Start frontend server
echo "ArtiQuest Frontend" 
cd frontend 
nohup npm run dev &
cd -

echo ===========================================
echo Application is starting...
echo ===========================================
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo You can now access the application in your browser.
sleep 1
