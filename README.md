# 🤟 ASL Vision

A real-time American Sign Language (ASL) detection web application. It uses a FastAPI Python backend to process webcam frames via WebSockets, and a React frontend to display live bounding boxes, skeletal tracking, and spelled sentences.

## Features
- **🔤 A-Z Letters:** Recognizes individual alphabet signs using a CNN model and heuristic refinement. Hold the sign steady to spell out words.
- **💬 Phrases:** Recognizes continuous dynamic signs (hello, thanks, iloveyou) using an LSTM model.
- **🤙 Gestures:** Instant detection of basic hand shapes (Open, Close, Pointer, OK) via TFLite.

---

## ⚡ Quick Start (Windows)
The easiest way to run the full application is to double-click the included batch file:
```text
start_project.bat
```
This automatically activates the backend environment, mounts the server, starts the frontend, and opens your browser.

---

## 🛠 Manual Setup & Running

If you prefer to run the components manually, follow these steps:

### 1. Backend (FastAPI + ML Models)

The backend runs on Python 3.10+ and serves both the models and the WebSocket endpoints on `localhost:8000`.

**Setup Environment:**
```powershell
# Create virtual environment (if it doesn't already exist)
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Run Server:**
```powershell
# From the project root, ensure you are in the virtual environment
.\venv\Scripts\activate
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```
*Note: The first startup takes ~10 seconds while the CNN, LSTM, and MediaPipe models load into memory.*

### 2. Frontend (React + Vite)

The frontend is a lightweight React app built visually with Vite. It runs on `localhost:5173`.

**Setup & Install:**
```powershell
cd frontend
npm install
```

**Run Development Server:**
```powershell
npm run dev
```

---

## 📁 Repository Structure
- `/backend/` — Contains FastAPI server, `action.h5`, `cnn8grps_rad1_model.h5`, geometric heuristics, and TFLite folders.
- `/frontend/` — React scaffolding, Tailwind/Vanilla CSS overlays, and UI.
- `requirements.txt` — Python dependencies needed for the backend.

## ⚙️ How it works
1. Your browser captures frames from your webcam and sends them over WebSocket to the FastAPI backend.
2. The backend runs OpenCV / CVZone to detect hand locations.
3. The Region Of Interest (hand crop) is passed to a Convolutional Neural Network (CNN) or LSTM model depending on the active mode in the UI.
4. Bounding boxes, coordinates, and predicted letters/signs are sent back as JSON.
5. The React frontend overlays this data dynamically on your live webcam feed.
