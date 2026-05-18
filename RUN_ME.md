RUN ME - Whale Project

Prerequisites
- Windows 10 or 11
- Python 3.10 or 3.11
- Node.js 20 or newer
- Git optional

Project Layout Used By These Steps
- Backend API: app.py
- Frontend: whales_website

Backend Setup And Run
1. Open PowerShell in project root.
2. Create and activate virtual environment:
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
3. Upgrade pip:
   python -m pip install --upgrade pip
4. Install backend dependencies:
   pip install fastapi uvicorn torch numpy scipy pillow python-multipart soundfile scikit-learn
5. Start backend server:
   python app.py
   - to confirm if the server is responding on port 8000:
   Invoke-WebRequest -Uri http://127.0.0.1:8000/ -UseBasicParsing | Select-Object -ExpandProperty Content

Expected backend URL
- http://127.0.0.1:8000

Quick backend health checks
- Browser:
  - http://127.0.0.1:8000/
  - http://127.0.0.1:8000/classes

Frontend Setup And Run
1. Open a second PowerShell terminal.
2. Go to frontend folder:
   cd whales_website
3. Install packages:
   npm install
4. Run dev server:
   npm run dev
5. Open the local URL shown in terminal, commonly:
   http://localhost:5173

Optional production frontend build
1. In whales_website:
   npm run build
2. Preview production build:
   npm run preview

Quick API Test With PowerShell
- Replace C:\path\to\your_audio.wav with a real audio file path.

$uri = "http://127.0.0.1:8000/predict"
$form = @{ file = Get-Item "C:\path\to\your_audio.wav" }
Invoke-RestMethod -Uri $uri -Method Post -Form $form

Common Issues
- Error about script execution policy when activating venv:
  Run PowerShell as current user and execute:
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
- Audio decoding fails:
  Ensure file is valid audio and retry with WAV.
- Frontend cannot reach backend:
  Ensure backend is running on port 8000 before opening frontend.
- Missing model files:
  Confirm .pth and .pkl files are in project root beside app.py.

What Not To Share
- .venv
- whales_website/node_modules
- __pycache__
- whales_website/dist

Minimum Share Bundle
- app.py
- whale_gate_classifier.pth
- whale_species_classifier.pth
- whale_species_label_encoder.pkl
- whales_website folder without node_modules and dist
- RUN_ME.md
