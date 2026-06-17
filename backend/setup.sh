#!/bin/bash

set -e

cd "$(dirname "$0")"

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

echo "==> Downloading MediaPipe pose model (one-time, ~6MB)..."
if [ -f "pose_landmarker.task" ]; then
  echo "    Model already exists, skipping download."
else
  curl -L -o pose_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
  echo "    Model downloaded."
fi

echo "==> Checking for ANTHROPIC_API_KEY..."
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "    WARNING: ANTHROPIC_API_KEY is not set in your environment."
  echo "    Set it with: export ANTHROPIC_API_KEY=sk-ant-..."
  echo "    (Get a key at https://console.anthropic.com)"
else
  echo "    Found."
fi

mkdir -p uploads outputs

echo ""
echo "Setup complete. Start the server with:"
echo "    uvicorn main:app --reload --port 8000"
