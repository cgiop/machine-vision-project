@echo off
title SignCraft Studio

echo ================================================================
echo   SignCraft Studio
echo   Personal Real-Time Sign Language Workspace
echo ================================================================
echo.
echo   MODES:
echo   [1] Alphabet  - CNN + heuristics for A-Z spelling
echo   [2] Phrase    - LSTM for hello / thanks / iloveyou
echo   [3] Gesture   - TFLite for quick hand cues
echo.
echo   Frontend  ^>  http://localhost:5173
echo   Backend   ^>  http://localhost:8000
echo   API Docs  ^>  http://localhost:8000/docs
echo ================================================================
echo.

echo [1/2] Starting FastAPI backend in the existing venv...
start "SignCraft Backend" cmd /k "cd /d %~dp0 && venv\Scripts\uvicorn.exe backend.server:app --reload --host 0.0.0.0 --port 8000"

echo Waiting 15s for models to load...
timeout /t 15 /nobreak > nul

echo [2/2] Starting Vite frontend...
start "SignCraft Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Both services launched.
echo Use the browser controls to switch modes and tune confidence or hold timing.
echo.
pause > nul
