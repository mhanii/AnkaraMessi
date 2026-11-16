#!/usr/bin/env bash
# Download the model weights the pipeline uses into ./models (git-ignored). About 450 MB.
# Licenses are not all clear; see "Models and licenses" in README.md before redistributing.
set -euo pipefail
mkdir -p models/tts
fetch() { [ -s "$2" ] || curl -L --fail --retry 3 -o "$2" "$1"; }

# People + ball detector (YOLO11n) and 29-landmark pitch model, by Adit Jain ("Soccana").
fetch https://huggingface.co/Adit-jain/soccana/resolve/main/Model/weights/best.pt models/soccana_det.pt
fetch https://huggingface.co/Adit-jain/Soccana_Keypoint/resolve/main/Model/weights/best.pt models/soccana_keypoints.pt

# Kokoro-82M text-to-speech, ONNX build.
fetch https://huggingface.co/fastrtc/kokoro-onnx/resolve/main/kokoro-v1.0.onnx models/tts/kokoro-v1.0.onnx
fetch https://huggingface.co/fastrtc/kokoro-onnx/resolve/main/voices-v1.0.bin models/tts/voices-v1.0.bin
ls -la models models/tts
