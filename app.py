"""FastAPI backend for whale-only prediction using a gate + species model pipeline."""

import io
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from scipy import signal

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except Exception:
    HAS_SOUNDFILE = False

# Configuration aligned with notebook preprocessing
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_WIDTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent

# Model artifacts from the two-stage notebook pipeline
GATE_MODEL_PATH = BASE_DIR / "whale_gate_classifier.pth"
SPECIES_MODEL_PATH = BASE_DIR / "whale_species_classifier.pth"
SPECIES_ENCODER_PATH = BASE_DIR / "whale_species_label_encoder.pkl"


class WhaleCNN(nn.Module):
    """CNN model used for both the gate and whale species classifier."""

    def __init__(self, num_classes: int):
        super().__init__()

        self.conv1a = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.20)

        self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.20)

        self.conv3a = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(0.25)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout(0.25)

        self.adaptive = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.bnf1 = nn.BatchNorm1d(128)
        self.df1 = nn.Dropout(0.35)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.drop1(self.pool1(x))

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.drop2(self.pool2(x))

        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.drop3(self.pool3(x))

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop4(self.pool4(x))

        x = self.adaptive(x)
        x = x.view(x.size(0), -1)
        x = self.df1(F.relu(self.bnf1(self.fc1(x))))
        x = self.fc2(x)
        return x


def create_mel_filterbank(sr, n_fft, n_mels, fmin=20, fmax=8000):
    if fmax is None:
        fmax = sr / 2.0

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    freqs = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1

        for k in range(left, center):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb


def compute_mel_spectrogram(audio, sr):
    noverlap = N_FFT - HOP_LENGTH

    _, _, sxx = signal.spectrogram(
        audio,
        fs=sr,
        window="hann",
        nperseg=N_FFT,
        noverlap=noverlap,
        nfft=N_FFT,
        scaling="density",
        mode="magnitude",
    )

    mel_fb = create_mel_filterbank(sr, N_FFT, N_MELS, fmin=20, fmax=8000)
    mel_spec = mel_fb @ (sxx ** 2)
    mel_db = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))

    floor = mel_db.max() - 80.0
    mel_db = np.maximum(mel_db, floor)
    mel_norm = ((mel_db - floor) / 80.0).astype(np.float32)

    if mel_norm.shape[1] != TARGET_WIDTH:
        img = Image.fromarray((mel_norm * 255).astype(np.uint8))
        img = img.resize((TARGET_WIDTH, N_MELS), Image.BILINEAR)
        mel_norm = np.array(img).astype(np.float32) / 255.0

    return mel_norm


def load_audio_from_bytes(audio_bytes: bytes):
    audio, sr = None, None

    if HAS_SOUNDFILE:
        try:
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            audio = audio.astype(np.float32)
        except Exception:
            audio, sr = None, None

    if audio is None:
        raise ValueError("Could not decode audio bytes. Please upload a valid audio file.")

    if sr != SAMPLE_RATE:
        audio = signal.resample_poly(audio, SAMPLE_RATE, sr)
        sr = SAMPLE_RATE

    max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0
    if max_val > 0:
        audio = audio / max_val

    return audio.astype(np.float32), sr


def extract_audio_analysis(audio_data, sr):
    duration_seconds = float(len(audio_data) / sr)
    nperseg = min(N_FFT, len(audio_data))
    noverlap = max(0, nperseg - HOP_LENGTH)

    freqs, _, sxx = signal.spectrogram(
        audio_data,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=N_FFT,
        scaling="density",
        mode="magnitude",
    )

    mean_spectrum = sxx.mean(axis=1) if sxx.size else np.array([0.0])
    dominant_idx = int(np.argmax(mean_spectrum)) if len(mean_spectrum) else 0
    dominant_frequency_hz = float(freqs[dominant_idx]) if len(freqs) else 0.0

    if np.sum(mean_spectrum) > 0 and len(freqs) > 0:
        spectral_centroid_hz = float(np.sum(freqs * mean_spectrum) / np.sum(mean_spectrum))
    else:
        spectral_centroid_hz = 0.0

    rms_energy = float(np.sqrt(np.mean(np.square(audio_data))))
    zero_crossing_rate = float(np.mean(np.abs(np.diff(np.sign(audio_data)))) / 2.0)

    if spectral_centroid_hz < 250:
        frequency_band = "low"
    elif spectral_centroid_hz < 1000:
        frequency_band = "mid"
    else:
        frequency_band = "high"

    return {
        "duration_seconds": round(duration_seconds, 2),
        "dominant_frequency_hz": round(dominant_frequency_hz, 2),
        "spectral_centroid_hz": round(spectral_centroid_hz, 2),
        "rms_energy": round(rms_energy, 6),
        "zero_crossing_rate": round(zero_crossing_rate, 6),
        "frequency_band": frequency_band,
    }


def audio_to_melspectrogram(audio_bytes: bytes):
    try:
        audio_data, sr = load_audio_from_bytes(audio_bytes)
        mel_spec = compute_mel_spectrogram(audio_data, sr)
        audio_analysis = extract_audio_analysis(audio_data, sr)
        return mel_spec, audio_analysis
    except Exception as exc:
        raise ValueError(f"Error processing audio: {exc}")


def _load_artifacts():
    if not GATE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing gate model file: {GATE_MODEL_PATH}")
    if not SPECIES_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing species model file: {SPECIES_MODEL_PATH}")
    if not SPECIES_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Missing species label encoder file: {SPECIES_ENCODER_PATH}")

    with open(SPECIES_ENCODER_PATH, "rb") as f:
        species_encoder = pickle.load(f)

    gate_ckpt = torch.load(GATE_MODEL_PATH, map_location=DEVICE, weights_only=False)
    species_ckpt = torch.load(SPECIES_MODEL_PATH, map_location=DEVICE, weights_only=False)

    gate_model = WhaleCNN(num_classes=2).to(DEVICE)
    gate_model.load_state_dict(gate_ckpt["model_state_dict"])
    gate_model.eval()

    species_model = WhaleCNN(num_classes=len(species_encoder.classes_)).to(DEVICE)
    species_model.load_state_dict(species_ckpt["model_state_dict"])
    species_model.eval()

    gate_class_names = gate_ckpt.get("class_names", ["Dolphin", "Whale"])
    gate_threshold = float(gate_ckpt.get("whale_gate_threshold", 0.60))

    whale_idx = 1
    for idx, name in enumerate(gate_class_names):
        if "whale" in str(name).lower():
            whale_idx = idx
            break

    return gate_model, species_model, species_encoder, gate_class_names, whale_idx, gate_threshold


gate_model, species_model, label_encoder, gate_class_names, gate_whale_idx, whale_gate_threshold = _load_artifacts()

app = FastAPI(title="Whale Audio Classification API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {
        "message": "Whale gate + species API is running",
        "gate_threshold": whale_gate_threshold,
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Handle audio file upload and return species prediction.
    
    Expected input: Audio file (WAV, MP3, etc.)
    Returns: JSON with predicted species, confidence, and all class probabilities
    """
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Convert to mel-spectrogram
        mel_spec, audio_analysis = audio_to_melspectrogram(contents)
        mel_tensor = torch.FloatTensor(mel_spec[np.newaxis, np.newaxis, :, :]).to(DEVICE)

        with torch.no_grad():
            gate_logits = gate_model(mel_tensor)
            gate_probs = F.softmax(gate_logits, dim=1).cpu().numpy()[0]

        whale_prob = float(gate_probs[gate_whale_idx])
        gate_pred_idx = int(np.argmax(gate_probs))
        gate_pred_label = str(gate_class_names[gate_pred_idx])

        if whale_prob < whale_gate_threshold:
            confidence = float(1.0 - whale_prob)
            return {
                "predicted_species": "Not_Whale",
                "confidence": confidence,
                "confidence_percent": f"{confidence * 100:.2f}%",
                "all_predictions": {},
                "audio_analysis": audio_analysis,
                "status": "success",
                "gate": {
                    "label": gate_pred_label,
                    "whale_probability": whale_prob,
                    "threshold": whale_gate_threshold,
                },
            }

        with torch.no_grad():
            species_logits = species_model(mel_tensor)
            species_probs = F.softmax(species_logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(species_probs))
        predicted_species = label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(species_probs[pred_idx])

        all_predictions = {
            class_name: float(species_probs[i])
            for i, class_name in enumerate(label_encoder.classes_)
        }

        return {
            "predicted_species": predicted_species,
            "confidence": confidence,
            "confidence_percent": f"{confidence * 100:.2f}%",
            "all_predictions": all_predictions,
            "audio_analysis": audio_analysis,
            "status": "success",
            "gate": {
                "label": gate_pred_label,
                "whale_probability": whale_prob,
                "threshold": whale_gate_threshold,
            },
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.get("/classes")
def get_classes():
    return {
        "classes": label_encoder.classes_.tolist(),
        "num_classes": len(label_encoder.classes_),
        "gate_classes": list(gate_class_names),
        "gate_threshold": whale_gate_threshold,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
