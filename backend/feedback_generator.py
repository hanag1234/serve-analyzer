"""
Turns extracted biomechanical metrics into plain-language coaching feedback
using the Claude API.

Reference ranges below are deliberately simple, directional heuristics meant
to flag things worth a coach's attention -- NOT a substitute for a real
biomechanics study. As you get real user footage, these should be tuned
against footage from coaches/players you trust.
"""

import os
import json
from anthropic import Anthropic


# Rough reference ranges for an adult recreational-to-competitive serve.
# These are intentionally generous bands, not hard pass/fail thresholds.
REFERENCE_RANGES = {
    "elbow_angle_at_contact_deg": (155, 180),       # near-full extension at contact
    "knee_bend_at_loading_deg": (110, 150),         # meaningful knee flexion at the load
    "shoulder_hip_separation_at_contact_deg": (10, 45),  # some trunk rotation differential
    "contact_height_rel_to_shoulder": (0.3, 1.2),   # contact comfortably above shoulder line
    "toss_peak_vs_contact_height_rel": (-0.3, 0.4), # toss peak roughly near/above contact height
}

METRIC_DESCRIPTIONS = {
    "elbow_angle_at_contact_deg": "Elbow extension angle at ball contact",
    "knee_bend_at_loading_deg": "Knee bend during the loading/trophy phase",
    "shoulder_hip_separation_at_contact_deg": "Shoulder-to-hip rotation separation at contact",
    "contact_height_rel_to_shoulder": "Contact point height relative to shoulder",
    "toss_peak_vs_contact_height_rel": "Toss peak height relative to contact point",
}


def _build_metrics_summary(metrics: dict) -> str:
    lines = []
    for key, value in metrics.items():
        desc = METRIC_DESCRIPTIONS.get(key, key)
        ref = REFERENCE_RANGES.get(key)
        ref_str = f" (typical range: {ref[0]}-{ref[1]})" if ref else ""
        lines.append(f"- {desc}: {value}{ref_str}")
    return "\n".join(lines)


SYSTEM_PROMPT = """You are an experienced tennis serve coach reviewing biomechanical \
data extracted from a video of a player's serve via pose estimation. You will be given \
a set of measured metrics at key phases of the serve (loading, contact), along with \
typical reference ranges for context.

Your job: write clear, encouraging, specific coaching feedback a real tennis coach would \
give after watching a serve. Rules:
- Lead with one genuine strength based on the metrics, even if subtle.
- Identify 1-3 specific areas to work on, each tied directly to a metric that falls \
outside or near the edge of its typical range.
- For each area, give one concrete, actionable drill or cue a player could use on-court \
THIS WEEK (not vague advice like "improve your rotation" -- something like "Practice the \
trophy position pause: serve in slow motion, holding the loaded position for 2 full \
seconds before swinging up").
- Keep the tone like a supportive coach, not a clinical report. No bullet-point jargon dumps.
- Do not claim certainty the data can't support -- pose estimation from a single phone \
camera has real limitations (especially for rotation/depth), so hedge appropriately on \
metrics like shoulder-hip separation.
- Keep total feedback to about 150-220 words.
- Do not use markdown headers. Short paragraphs and at most one short list are fine.
"""


def generate_feedback(metrics: dict, dominant_side: str, api_key: str = None) -> str:
    """
    Calls Claude with the extracted metrics and returns natural-language
    coaching feedback as a string.
    """
    if not metrics:
        return (
            "We weren't able to confidently extract enough body-tracking data from "
            "this clip to generate feedback. Try filming again with the full body "
            "clearly visible from a side-on angle, good lighting, and the camera "
            "held steady."
        )

    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    metrics_summary = _build_metrics_summary(metrics)
    user_message = (
        f"Hitting arm: {dominant_side}\n\n"
        f"Measured metrics:\n{metrics_summary}\n\n"
        "Please provide coaching feedback based on this data."
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text_blocks = [block.text for block in response.content if block.type == "text"]
    return "\n".join(text_blocks).strip()


def metrics_to_json(metrics: dict, dominant_side: str, phase_frames: dict) -> str:
    """Helper for returning a clean JSON payload to the frontend alongside the text feedback."""
    return json.dumps({
        "dominant_side": dominant_side,
        "metrics": metrics,
        "reference_ranges": {k: REFERENCE_RANGES[k] for k in metrics if k in REFERENCE_RANGES},
        "phase_frames": phase_frames,
    }, indent=2)
