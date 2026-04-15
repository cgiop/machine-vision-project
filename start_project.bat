@echo off
title ASL Vision Core v5.0 — 4-Engine System

echo ================================================================
echo   ASL Vision Core v5.0
echo   Real-Time Sign Language Detection
echo ================================================================
echo.
echo   DETECTION MODES:
echo   [1] Phrase Mode     — LSTM: hello / thanks / iloveyou
echo   [2] Gesture Mode    — TFLite (kinivi): Open/Close/Pointer/OK
echo   [3] Google MP Mode  — GestureRecognizer: 7 gestures + ILoveYou
echo   [4] Alphabet Mode   — CNN A-Z: full 26-letter ASL alphabet
echo.
echo   Frontend  ^>  http://localhost:5173
echo   Backend   ^>  http://localhost:8000
echo   API Docs  ^>  http://localhost:8000/docs
echo ================================================================
echo.

echo [1/2] Starting FastAPI backend (all 4 models loading, ~15s wait)...
start "ASL Backend v5.0" cmd /k "cd /d %~dp0 && venv\Scripts\uvicorn.exe backend.server:app --reload --host 0.0.0.0 --port 8000"

echo Waiting 15s for models to load...
timeout /t 15 /nobreak > nul

echo [2/2] Starting Vite frontend...
start "ASL Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Both services launched!
echo Switch modes in the top bar: Phrase / Gesture / Google MP / A-Z Alphabet
echo.
pause > nul
