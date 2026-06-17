# Serve Lab — AI Tennis Serve Analyzer

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
│   ├── feedback_generator.py   Calls Claude to turn metrics into coaching feedback
│   ├── requirements.txt        Python dependencies
│   ├── setup.sh                One-time setup 
│   └── pose_landmarker.task    
├── frontend/
│   └── index.html              Single-page upload + results UI 
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

You should see something like `Uvicorn running on http://127.0.0.1:8000`.
Visit `http://localhost:8000/health` in a browser — you should see `{"status":"ok"}`.

### 3. Open the frontend

Just open `frontend/index.html` directly in your browser (double-click it, or
`open frontend/index.html` on Mac). No build step, no npm install — it's a
single static HTML file with vanilla JS.

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

The single biggest factor in result quality is camera angle and framing, not
code. For the MVP:

- **Camera position:** directly behind the server, roughly at baseline height
  (waist-to-shoulder height off the ground), 15-25 feet back. A phone propped
  on a fence or tripod works fine.
- **Framing:** full body in frame for the ENTIRE motion — from the toss starting
  to well after contact. If the racket or toss arm leaves the frame at any point,
  that metric will be unreliable.
- **One serve per clip.** Don't send a clip with multiple serves — phase detection
  assumes one loading → contact cycle per video.
- **Lighting:** outdoor daylight or well-lit indoor courts. MediaPipe struggles
  with silhouettes (backlighting) and very low light.
- **Clip length:** 2-5 seconds is ideal. Trim before uploading if your camera
  captured a longer rally.

Side-on or elevated (umpire-chair-height) footage will run through the pipeline
without errors, but the metrics — especially shoulder-hip separation and contact
height — will be less reliable, since the phase-detection logic assumes a
roughly front-facing view of the hitting arm's vertical motion.

---

## What this can and can't do yet (read before showing anyone)

**What it's good at right now:**
- Tracking gross body position (where the wrists, knees, hips are) reliably on
  clear, well-framed video.
- Giving directionally useful feedback on toss height, knee bend, and elbow
  extension — these are well-supported by a single 2D camera angle.
- Producing a genuinely nice-looking annotated video with skeleton + phase labels,
  which is itself useful for a coach to look at frame-by-frame even before reading
  any text.

**What it's NOT good at yet, and you should say so if anyone asks:**
- **Shoulder rotation / internal rotation** is fundamentally hard to measure from
  a single 2D camera — true rotation happens in the depth axis, which a single
  camera can't see well. The `shoulder_hip_separation_at_contact_deg` metric is a
  rough proxy, not a precise measurement. The feedback prompt is written to hedge
  on this, but don't oversell it.
- **Phase detection is heuristic, not validated.** It finds "contact" by looking
  for the highest point of the hitting wrist after a quarter of the clip has
  passed. This works well on clean footage but can misfire on unusual serve
  motions (e.g. a kick serve with an unusually high arc, or a player who tosses
  very low). You should manually sanity-check the labeled annotated video against
  a handful of real serves before trusting the metrics blindly.
- **Reference ranges are placeholders.** The "typical range" numbers in
  `feedback_generator.py` are reasonable starting heuristics, not numbers pulled
  from a real biomechanics study or validated against your own footage. Before
  you'd want to charge anyone money for this, you should validate these ranges
  against footage of players/serves you trust (ideally including some labeled by
  a real coach), and adjust.
- **No racket tracking at all.** This only tracks the body, not the racket head.
  Racket-specific things (string-bed angle at contact, racket head speed) are out
  of scope for this MVP.
- **Single person in frame assumed.** If a coach or another player is visible in
  the background, pose detection may occasionally lock onto the wrong person.

None of this means the MVP isn't useful — it means be honest about it being v1
when you show it to coaches, and treat their reactions as the validation signal
for whether to keep building.

---

## Suggested next steps, in order

1. **Test on 10-15 real serves** from people you know (yourself, club members).
   Watch the annotated video for each and sanity-check: did it actually label
   the right frame as "contact"? Does the knee-bend number look plausible?
2. **Show it to one real coach** (Sharon Heights or Olympic Club, since you
   already have that relationship) before building anything else. Their reaction
   to the feedback text quality matters more than any code improvement right now.
3. **Tune reference ranges** based on what real coaches say looks "good" vs.
   "needs work" on the footage you collect.
4. **Improve phase detection robustness** once you've seen real failure cases —
   you'll likely want to add a sanity check (e.g. "racket-side wrist must be the
   highest point in a specific y-range relative to the head") rather than the
   current pure local-min/max approach.
5. **Only then** consider a real mobile app, accounts/auth, payment, multi-serve
   clips, or anything else that adds engineering surface area without adding
   validation.

---

## API reference (for when you want to build a real frontend later)

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
- Each analysis makes one Claude API call (~600 tokens out) — cost is a fraction
  of a cent per analysis at current pricing, but check console.anthropic.com for
  current rates.
- No other paid services are used in this MVP.
