@echo off
title Drishti 2.5D Launcher
echo ========================================================
echo Launching Drishti 2.5D Full Stack (Backend + Frontend)...
echo ========================================================

start "Drishti Backend (Port 8765)" cmd /k "cd /d "%~dp0Backend" && "E:\anaconda\python.exe" -m backend.main"
timeout /t 2 /nobreak >nul
start "Drishti Frontend (Port 5173)" cmd /k "cd /d "%~dp0Frontend" && npm run dev"

echo Services started in separate terminal windows.
echo Frontend URL: http://localhost:5173
echo Backend WebSocket: ws://127.0.0.1:8765
