"""
backend/main.py
FastAPI server: เสิร์ฟหน้าเว็บ frontend + WebSocket endpoint สำหรับรับเฟรมจากกล้องฝั่ง browser
ประมวลผลด้วย MediaPipe + โมเดลที่เทรนไว้ (model.pkl) แล้วส่งผลทำนาย (ซับ) กลับไปแบบเรียลไทม์
 
รัน:
    uvicorn backend.main:app --reload --port 8000
แล้วเปิดเบราว์เซอร์ที่ http://localhost:8000
"""
 
import base64
import json
import os
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.features import landmarks_to_feature_vector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
 
mp_hands = mp.solutions.hands
 
# ----- ปรับค่า smoothing/threshold ได้ตรงนี้ -----
SMOOTHING_WINDOW = 10
STABLE_RATIO = 0.7
CONF_THRESHOLD = 0.55
IGNORE_LABEL = "no_sign"
MAX_SENTENCE_WORDS = 8
# --------------------------------------------------
 
app = FastAPI(title="TSL Signal Decoder")
 
_model_bundle = None
if os.path.exists(MODEL_PATH):
    _model_bundle = joblib.load(MODEL_PATH)
 
 
@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
 
 
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
 
 
@app.get("/api/status")
def status():
    """ให้ frontend เช็คได้ว่าโมเดลพร้อมใช้งานไหม และรู้จัก label อะไรบ้าง"""
    if _model_bundle is None:
        return {"model_loaded": False, "labels": []}
    return {"model_loaded": True, "labels": _model_bundle["labels"]}
 
 
def decode_frame(frame_b64: str):
    """แปลง dataURL (base64 jpeg) จาก browser -> ndarray BGR สำหรับ OpenCV"""
    header_free = frame_b64.split(",")[-1]
    img_bytes = base64.b64decode(header_free)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
 
 
@app.websocket("/ws/predict")
async def predict_ws(websocket: WebSocket):
    await websocket.accept()
 
    if _model_bundle is None:
        await websocket.send_json(
            {"error": "ยังไม่พบโมเดล (model.pkl) — รัน train_model.py ก่อน"}
        )
        await websocket.close()
        return
 
    clf = _model_bundle["model"]
 
    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
    sentence_words = deque(maxlen=MAX_SENTENCE_WORDS)
    last_committed_label = None
 
    # แก้ไข max_num_hands เป็น 2 ให้ตรงกับระบบที่เทรนมา
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
                    {"current": "", "sentence": "", "confidence": 0.0, "hand_detected": False}
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
                # แก้ไขการส่งข้อมูลเป็นแบบ 2 มือ (Zero Padding)
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
                if stable_label != IGNORE_LABEL and stable_label != last_committed_label:
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