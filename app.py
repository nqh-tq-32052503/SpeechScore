from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import uvicorn
from scoring import Scoring
import os 
import tempfile
import soundfile as sf
import librosa
import numpy as np
import resampy


app = FastAPI(title="STT API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
SAMPLE_RATE = 16000

# TODO: load your model/decoder once at startup
@app.on_event("startup")
def _load():
    # Example:
    # global asr
    # asr = YourASR.load_from_checkpoint("/models/ckpt.pt")
    global MODEL
    MODEL = Scoring()

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

def convert_to_mono_16k_wav(input_path: str, output_path: str):
    """
    Convert an audio file of any format to a mono, 16 kHz WAV file.
    
    Args:
        input_path (str): Path to the input audio (any format readable by librosa or soundfile).
        output_path (str): Path to save the converted audio (should end with .wav).
    """
    # --- Load audio (librosa handles most formats via ffmpeg) ---
    audio, sr = librosa.load(input_path, sr=None, mono=False)
    # sr=None preserves original sample rate

    # --- Convert to mono if stereo or multi-channel ---
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # --- Resample to 16 kHz if needed ---
    target_sr = 16000
    if sr != target_sr:
        audio = resampy.resample(audio, sr, target_sr)
        sr = target_sr

    # --- Normalize to float32 for safety ---
    audio = np.asarray(audio, dtype=np.float32)

    # --- Ensure output directory exists ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Save as WAV ---
    sf.write(output_path, audio, sr, subtype="PCM_16")

    print(f"✅ Converted {input_path} → {output_path} [mono, 16 kHz WAV]")

@app.post("/inference")
def inference(file: UploadFile = File(...)):
    # Save upload to temp, convert to wav16k if needed
    with tempfile.TemporaryDirectory() as td:
        src_path = os.path.join(td, file.filename)
        with open(src_path, "wb") as f:
            f.write(file.file.read())

        wav_path = os.path.join(td, "pcm16k.wav")
        print("Converting input to 16k WAV...")
        try:
            convert_to_mono_16k_wav(src_path, wav_path)
        except Exception as e:
            raise HTTPException(400, f"ffmpeg failed to decode input: {e}")
        print("Converted input to 16k WAV")
        assessment_result = MODEL.inference(wav_path, window_size=None)
        return assessment_result

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8003, reload=True)