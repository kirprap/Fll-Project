@echo off
echo Starting ArtiQuest Development Environment
echo.

echo Starting FastAPI backend...
start "ArtiQuest Backend" cmd /k "cd MainApp\backend && python main.py"

timeout /t 3 /nobreak >nul

echo Starting React frontend...
start "ArtiQuest Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Both servers are starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
pause
