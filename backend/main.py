"""
FastAPI server for the serve analyzer MVP.

Endpoints:
  POST /analyze   -> upload a video, get back metrics + coaching feedback + annotated video URL
  GET  /videos/{filename} -> serve the annotated video file

Run locally with:
  uvicorn main:app --reload --port 8000
"""

import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from pose_analyzer import analyze_video
from feedback_generator import generate_feedback

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Serve Analyzer API")

# Allow the frontend (running on a different port/origin during dev) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real frontend domain before going live
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}
MAX_FILE_SIZE_MB = 100


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    job_id = str(uuid.uuid4())[:8]
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    output_filename = f"{job_id}_annotated.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Save uploaded file to disk
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    size_mb = os.path.getsize(input_path) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        os.remove(input_path)
        raise HTTPException(status_code=400, detail=f"File too large ({size_mb:.1f}MB). Max {MAX_FILE_SIZE_MB}MB.")

    try:
        result = analyze_video(input_path, output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {e}")
    finally:
        # Clean up the raw upload once we're done with it
        if os.path.exists(input_path):
            os.remove(input_path)

    if not result.metrics:
        feedback_text = generate_feedback({}, result.dominant_side)
    else:
        try:
            feedback_text = generate_feedback(result.metrics, result.dominant_side)
        except Exception as e:
            feedback_text = (
                "Metrics were extracted successfully, but we couldn't generate written "
                f"feedback right now ({e}). See the metrics below for the raw numbers."
            )

    return JSONResponse({
        "job_id": job_id,
        "dominant_side": result.dominant_side,
        "metrics": result.metrics,
        "phase_frames": {
            "loading": result.loading_frame_idx,
            "racket_drop": result.racket_drop_frame_idx,
            "contact": result.contact_frame_idx,
        },
        "fps": result.fps,
        "feedback": feedback_text,
        "annotated_video_url": f"/videos/{output_filename}",
    })


@app.get("/videos/{filename}")
def get_video(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4")
