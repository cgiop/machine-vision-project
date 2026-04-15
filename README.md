# SignCraft Studio

SignCraft Studio is a real-time American Sign Language recognition workspace built around your existing Python 3.10 virtual environment. It combines a FastAPI backend with a React frontend, but now presents as a single cohesive project with its own branding, tuning controls, and cleaner session behavior.

## What makes this version yours

- A new product identity and interface instead of generic merged-project styling.
- Live backend calibration from the frontend for confidence threshold and alphabet hold timing.
- Session-aware clear handling so the active transcript can be reset without restarting the socket.
- More stable transcript behavior, including reduced accidental repeated commits when the same sign stays in frame.
- Unified launch script and project copy that match the actual three supported modes.

## Supported modes

- `Alphabet`: CNN + gesture heuristics for A-Z spelling.
- `Phrase`: LSTM sequence model for `hello`, `thanks`, and `iloveyou`.
- `Gesture`: TFLite hand-shape classifier for quick gesture recognition.

## Quick start on Windows

If your `venv` is already created and populated, use:

```text
start_project.bat
```

That starts:

- Backend at `http://localhost:8000`
- Frontend at `http://localhost:5173`
- FastAPI docs at `http://localhost:8000/docs`

## Manual run

### Backend

```powershell
.\venv\Scripts\activate
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```powershell
cd frontend
npm run dev
```

## GitHub Codespaces without Docker

You should not upload or reuse the local `venv` in Codespaces because the local environment is Windows-based while Codespaces is Linux-based.

Instead, use the repo-managed Python 3.10 environment:

```bash
chmod +x scripts/setup_codespaces.sh
./scripts/setup_codespaces.sh
```

Then run:

```bash
~/.local/bin/micromamba run -n signcraft uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```

```bash
~/.local/bin/micromamba run -n signcraft npm --prefix frontend run dev -- --host 0.0.0.0
```

This creates a Linux-safe Python 3.10 environment in Codespaces without Docker.

## Notes

- The backend is intended to run inside the existing Python 3.10 virtual environment.
- The first backend startup can take a little time because TensorFlow and MediaPipe models are loaded into memory.
- The frontend can be pointed at a different backend by setting `VITE_API_URL` and optionally `VITE_WS_URL`.
