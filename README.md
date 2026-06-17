# AI Tennis Serve Analyzer

Upload a video of a tennis serve. Get pose tracked skeleton overlay, biomechanical
metrics at the key phases of the serve, and AI-generated coaching feedback.

This is a working MVP built to validate the idea fast, not a finished product.
Read the "What this can and can't do yet" section.

---

## How it works

1. You upload a video clip of someone serving, filmed from behind, full body in frame.
2. The backend runs MediaPipe's pose estimation model on every frame to track 33
   body keypoints (shoulders, elbows, wrists, hips, knees, ankles).
3. Simple logic finds three key moments in the serve: the loading/trophy position,
   the racket drop, and ball contact — based on how the hitting wrist moves up and down.
4. At those key moments, the code computes real biomechanical numbers: elbow angle
   at contact, knee bend at the load, shoulder-hip rotation separation, contact height,
   and toss height relative to contact.
5. Those numbers get sent to Claude with a coaching focused prompt, which writes
   feedback — a strength, 1-3 specific things to work on, and a concrete drill for each.
7. You get back an annotated video (skeleton drawn on, key frames labeled) plus the
   metrics and the written feedback.

---

## Project structure

```
serve-analyzer/
├── backend/
│   ├── main.py                 FastAPI server 
│   ├── pose_analyzer.py        Core CV logic: pose tracking, phase detection, metrics
│   ├── feedback_generator.py   Calls Claude
│   ├── requirements.txt        
│   ├── setup.sh                
│   └── pose_landmarker.task    
├── frontend/
│   └── index.html              
└── sample_data/
    └── (put test videos here)
```

---

## Setup — do this once

### 1. Backend

You'll need Python 3.10+ and an Anthropic API key (console.anthropic.com).

```bash
cd backend
bash setup.sh
export ANTHROPIC_API_KEY=sk-ant-...   
```

`setup.sh` installs the Python packages and downloads the MediaPipe pose model

### 2. Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 3. Open the frontend

Just open `frontend/index.html` 

If your backend is running anywhere other than `http://localhost:8000`, set
this at the top of your browser console or edit the file directly:
```js
window.SERVE_LAB_API_BASE = "http://your-backend-url:8000";
```

### 4. Try it

Film a serve (see filming guidance below), upload it, click "Analyze serve."
First run will be a bit slower since MediaPipe initializes; after that it's fast.

---

## Filming guidance (important for accuracy)

- **Framing:** full body in frame for the ENTIRE motion — from the toss starting
  to well after contact. 
- **One serve per clip.** Don't send a clip with multiple serves — phase detection
  assumes one loading → contact cycle per video.
- **Lighting:** outdoor daylight or well-lit indoor courts. MediaPipe struggles
  with silhouettes and very low light.
- **Clip length:** 2-5 seconds is ideal. Trim before uploading if your camera
  captured a longer rally.

Side-on or elevated (umpire-chair-height) footage will run through the pipeline
without errors, but the metrics — especially shoulder-hip separation and contact
height — will be less reliable, since the phase-detection logic assumes a
roughly front-facing view of the hitting arm's vertical motion.

---

## What this can and can't do yet 

**What it's good at right now:**
- Tracking gross body position (where the wrists, knees, hips are) reliably on
  clear, well-framed video.
- Giving directionally useful feedback on toss height, knee bend, and elbow
  extension.
- Producing a genuinely nice looking annotated video with skeleton + phase labels,
  which is itself useful for a coach to look at frame-by-frame even before reading
  any text.

**What it's NOT good at yet:**
- **Shoulder rotation / internal rotation** is fundamentally hard to measure from
  a single 2D camera. The `shoulder_hip_separation_at_contact_deg` metric is a
  rough proxy.
- **Phase detection is heuristic.** It finds "contact" by looking
  for the highest point of the hitting wrist after a quarter of the clip has
  passed. 
- **No racket tracking at all.** This only tracks the body, not the racket head.
  Racket-specific things (string-bed angle at contact, racket head speed) are out
  of scope for this MVP.

---

### `POST /analyze`
Multipart form upload, field name `file`. Accepts `.mp4`, `.mov`, `.m4v`, `.avi`,
max 100MB.

Response:
```json
{
  "job_id": "a1b2c3d4",
  "dominant_side": "right",
  "metrics": {
    "elbow_angle_at_contact_deg": 168.2,
    "knee_angle_at_contact_deg": 162.5,
    "knee_bend_at_loading_deg": 128.4,
    "shoulder_hip_separation_at_contact_deg": 22.1,
    "contact_height_rel_to_shoulder": 0.65,
    "toss_peak_vs_contact_height_rel": 0.1
  },
  "phase_frames": { "loading": 14, "racket_drop": 19, "contact": 24 },
  "fps": 30.0,
  "feedback": "Your toss height relative to contact point looks solid...",
  "annotated_video_url": "/videos/a1b2c3d4_annotated.mp4"
}
```

### `GET /videos/{filename}`
Serves the annotated MP4.

### `GET /health`
Returns `{"status": "ok"}` — useful for confirming the server is up.

---

## Costs to be aware of

- MediaPipe and OpenCV are free, run locally, no API costs.
- Each analysis makes one Claude API call (~600 tokens out)
- No other paid services are used in this MVP.
