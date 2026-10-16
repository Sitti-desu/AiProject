# Thai Sign Language Real-Time Translator

Real-time Thai Sign Language (TSL) hand-sign recognition that turns webcam video into live subtitles in the browser.

> Personal project · Python 3.11 · MediaPipe + scikit-learn + FastAPI

![demo](assets/demo.gif)

## Problem

Sign language recognition is usually framed as a deep-learning / video-classification problem, which is heavy to train and slow to run on a laptop CPU. This project instead recognizes static hand signs from MediaPipe's 21 hand landmarks: each landmark set is normalized (translated to the wrist, scaled by the wrist-to-middle-finger-MCP distance) into a fixed 63-dimensional feature vector that is invariant to where the hand is in the frame and how close it is to the camera. A classical RandomForest classifier on top of that vector is enough to get accurate, low-latency predictions without a GPU.

The other non-obvious problem is false positives when the hand is simply resting in frame between signs — solved by training an explicit `no_sign` class and smoothing predictions over a rolling window before committing a word to the sentence.

## Architecture

```
Browser webcam
      |  (canvas captures frame, sends as base64 JPEG)
      v
FastAPI WebSocket (/ws/predict)
      |
      v
MediaPipe Hands  --------->  21 hand landmarks (x, y, z)
      |
      v
Feature extraction  ------->  63-dim vector (wrist-relative, scale-normalized)
      |
      v
RandomForest classifier  --->  predicted label + confidence
      |
      v
Rolling-window smoothing  --->  stable label -> appended to sentence
      |
      v
WebSocket push  ----------->  Browser subtitle bar
```

A CLI equivalent (`scripts/translate_realtime.py`) runs the same MediaPipe -> features -> RandomForest pipeline locally with an OpenCV window instead of a browser.

## Results

Measured on the current dataset (`data/gestures.csv`), 1485 samples across 14 classes:

| Class | Samples |
|---|---|
| Cry | 219 |
| Dont know | 200 |
| why | 195 |
| I LOVE YOU | 192 |
| no_sign | 167 |
| 100 | 104 |
| mini heart | 75 |
| o | 61 |
| L | 50 |
| Y | 50 |
| u | 50 |
| I | 42 |
| E | 41 |
| v | 39 |

| Metric | Value |
|---|---|
| Dataset size | 1485 samples, 14 classes |
| Validation accuracy (temporal split) | 0.936 |
| Macro F1 (temporal split) | 0.952 |
| Validation accuracy (random split) | 0.997 |
| Macro F1 (random split) | 0.997 |
| Average inference latency (full path, live webcam) | 46.83 ms/frame |
| P95 inference latency | 52.79 ms/frame |
| Achievable FPS | 21.4 |

Accuracy is reported primarily on a **temporal split** (first 80% of each class's samples, in collection order, for training; last 20% for testing) rather than a random split. Samples collected seconds apart are near-duplicates (the hand barely moves between consecutive captures), so a random split can leak near-identical frames into both train and test and inflate accuracy. The temporal split evaluates on frames the model has not seen in any form during training, which is a more honest estimate — see `scripts/train_model.py` for both numbers side by side.

The gap between the two splits (0.997 random vs 0.936 temporal) is a useful result in its own right, not just noise: it exposes a real confusion the random split was hiding. Under the temporal split, `no_sign` recall drops to 0.58 — 14 of 33 test samples are misclassified as `100`. That is the exact failure mode observed during live use of the web app, where the model repeatedly reports `100` while the hand is simply resting in frame. The random split's 0.997 accuracy never surfaced this because near-duplicate `no_sign` frames from the same collection burst ended up on both sides of that split.

Inference latency was measured with `scripts/benchmark_inference.py`, timing the exact path used by the backend (frame decode -> MediaPipe hand detection -> feature extraction -> RandomForest predict) against a live webcam feed.

## Limitations

- All samples were collected from a single signer, one camera, and one
  lighting condition. The temporal split rules out frame-level leakage,
  but not the possibility that the model has simply learned this specific
  setup. Accuracy on an unseen signer or environment is untested and
  would almost certainly be lower.
- `no_sign` and `100` overlap in feature space: the temporal split shows
  `no_sign` frames get misclassified as `100` in roughly 40% of held-out
  cases. The two classes need more deliberately collected boundary
  samples to separate cleanly.
- Static hand shapes only. Signs that depend on motion, facial expression,
  or two-hand interaction are out of scope.

Next step: collect additional `no_sign` and `100` samples specifically at
the boundary between the two (loose fists, resting hand positions that
partially resemble `100`) to fix the confusion above. After that, collect
data across multiple sessions with a `session_id` column and evaluate
with a session-level split (train on sessions 1-3, test on session 4),
which measures generalization to unseen conditions rather than unseen
frames.

## Setup & Run

Requires Python 3.11 (MediaPipe does not support newer Python versions).

```
py -3.11 -m pip install -r requirements.txt

# collect training data for your own signs
py -3.11 scripts/collect_data.py

# train the model
py -3.11 scripts/train_model.py

# run the web app
py -3.11 -m uvicorn backend.main:app --reload --port 8000
# open http://localhost:8000

# or run the CLI version instead
py -3.11 scripts/translate_realtime.py
```

## Project structure

```
backend/
  main.py                 FastAPI server + WebSocket prediction endpoint
  features.py              landmarks -> 63-dim feature vector
scripts/
  collect_data.py          camera-based data collection tool
  train_model.py           trains RandomForest, reports metrics
  translate_realtime.py    CLI real-time translator (OpenCV window)
  benchmark_inference.py   measures live inference latency/FPS
frontend/
  index.html, app.js, style.css   browser UI
data/
  gestures.csv             collected training samples
models/
  model.pkl                trained classifier + label list
```
