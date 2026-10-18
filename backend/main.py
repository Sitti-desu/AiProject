"""
backend/main.py
FastAPI server: เสิร์ฟหน้าเว็บ frontend + WebSocket endpoint สำหรับรับเฟรมสด
พร้อม API บันทึกข้อมูลท่าทางลง CSV และสั่งเทรนโมเดลใหม่ผ่านหน้าเว็บ
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import base64
import csv
import json
import os
import sys
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.features import feature_names, landmarks_to_feature_vector
from scripts.train_model import train_model_from_csv

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
CSV_PATH = os.path.join(BASE_DIR, "data", "gestures.csv")

mp_hands = mp.solutions.hands

# ----- ปรับค่า smoothing/threshold -----
SMOOTHING_WINDOW = 10
STABLE_RATIO = 0.7
CONF_THRESHOLD = 0.55
IGNORE_LABEL = "no_sign"
MAX_SENTENCE_WORDS = 8
# ---------------------------------------

app = FastAPI(title="TSL Signal Decoder")

_model_bundle = None
if os.path.exists(MODEL_PATH):
    try:
        _model_bundle = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Warning: โหลด model.pkl ไม่สำเร็จ ({e})")


def decode_frame(frame_b64: str):
    try:
        header_free = frame_b64.split(",")[-1]
        img_bytes = base64.b64decode(header_free)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ---------------- API ENDPOINTS ----------------

@app.get("/api/status")
def status():
    """ตรวจสอบสถานะความพร้อมของโมเดล"""
    if _model_bundle is None:
        return {"model_loaded": False, "labels": []}
    labels = _model_bundle.get("labels", [])
    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    return {"model_loaded": True, "labels": list(labels)}


class CollectFrameRequest(BaseModel):
    label: str
    frame: str


@app.post("/api/collect")
def collect_sample(req: CollectFrameRequest):
    """บันทึกพิกัดมือลงใน data/gestures.csv"""
    image = decode_frame(req.frame)
    if image is None:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "ถอดรหัสรูปภาพไม่สำเร็จ"}
        )

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    ) as detector:
        results = detector.process(rgb)

    if not results.multi_hand_landmarks:
        return {"status": "no_hand", "message": "ตรวจไม่พบมือในเฟรมภาพ"}

    vec = landmarks_to_feature_vector(results)

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    file_exists = os.path.exists(CSV_PATH)

    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["label"] + feature_names)
        writer.writerow([req.label] + list(vec))

    return {"status": "success", "message": f"บันทึกท่า '{req.label}' เรียบร้อย"}


@app.post("/api/train")
def run_train():
    """สั่งรันเทรนโมเดล RandomForest จาก CSV ใหม่ทันที"""
    global _model_bundle
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        if not os.path.exists(CSV_PATH):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "ไม่พบไฟล์ data/gestures.csv กรุณาบันทึกท่าทางก่อนสั่งเทรน"}
            )

        labels = train_model_from_csv(csv_path=CSV_PATH, output_path=MODEL_PATH)

        if os.path.exists(MODEL_PATH):
            _model_bundle = joblib.load(MODEL_PATH)

        labels_list = list(labels) if labels is not None else []
        if hasattr(labels_list, "tolist"):
            labels_list = labels_list.tolist()

        return {
            "status": "success",
            "message": "เทรนโมเดลสำเร็จและอัปเดตระบบแล้ว",
            "labels": labels_list,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"เทรนโมเดลไม่สำเร็จ: {str(e)}"}
        )


# ---------------- WEBSOCKET ENDPOINT ----------------

@app.websocket("/ws/predict")
async def predict_ws(websocket: WebSocket):
    await websocket.accept()

    if _model_bundle is None:
        await websocket.send_json(
            {"error": "ยังไม่พบโมเดล (model.pkl) — กรุณาสั่งเทรนโมเดลก่อน"}
        )
        await websocket.close()
        return

    clf = _model_bundle["model"]
    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
    sentence_words = deque(maxlen=MAX_SENTENCE_WORDS)
    last_committed_label = None

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "clear":
                sentence_words.clear()
                last_committed_label = None
                recent_predictions.clear()
                await websocket.send_json(
                    {
                        "current": "",
                        "sentence": "",
                        "confidence": 0.0,
                        "hand_detected": False,
                    }
                )
                continue

            frame_b64 = msg.get("frame")
            if not frame_b64:
                continue

            image = decode_frame(frame_b64)
            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            predicted_label = None
            confidence = 0.0
            hand_detected = bool(results.multi_hand_landmarks)

            if hand_detected:
                vec = np.array(landmarks_to_feature_vector(results)).reshape(1, -1)
                probs = clf.predict_proba(vec)[0]
                best_idx = int(np.argmax(probs))
                best_label = clf.classes_[best_idx]
                confidence = float(probs[best_idx])
                if confidence >= CONF_THRESHOLD:
                    predicted_label = best_label

            recent_predictions.append(predicted_label)
            valid = [p for p in recent_predictions if p is not None]

            stable_label = None
            if len(valid) >= max(3, int(SMOOTHING_WINDOW * 0.5)):
                label, cnt = Counter(valid).most_common(1)[0]
                if cnt / len(recent_predictions) >= STABLE_RATIO:
                    stable_label = label

            current_display = ""
            if stable_label is not None:
                current_display = stable_label
                if (
                    stable_label != IGNORE_LABEL
                    and stable_label != last_committed_label
                ):
                    sentence_words.append(stable_label)
                    last_committed_label = stable_label
            elif not valid:
                last_committed_label = None

            await websocket.send_json(
                {
                    "current": current_display,
                    "sentence": " ".join(sentence_words),
                    "confidence": confidence,
                    "hand_detected": hand_detected,
                }
            )
    except WebSocketDisconnect:
        pass
    finally:
        hands.close()


# ---------------- STATIC & FRONTEND ROUTING ----------------

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/{full_path:path}")
def serve_spa(full_path: str):
    """ส่ง index.html สำหรับทุก Path ที่ไม่ใช่ API หรือ Static Files"""
    file_path = os.path.join(FRONTEND_DIR, full_path)
    if full_path and os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))