@echo off
title Drishti 2.5D Backend
echo ========================================================
echo Starting Drishti 2.5D Backend Perception Pipeline...
echo ========================================================
cd /d "%~dp0Backend"
"E:\anaconda\python.exe" -m backend.main %*
pause
