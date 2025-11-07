from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import uvicorn
from scoring import Scoring
import os 
import tempfile

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

@app.post("/inference")
def inference(file: UploadFile = File(...)):
    # Save upload to temp, convert to wav16k if needed
    with tempfile.TemporaryDirectory() as td:
        src_path = os.path.join(td, file.filename)
        with open(src_path, "wb") as f:
            f.write(file.file.read())
        assessment_result = MODEL.inference(src_path, window_size=5)
        return assessment_result

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8003, reload=True)