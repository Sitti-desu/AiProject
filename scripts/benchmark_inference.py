"""
benchmark_inference.py
วัด latency จริงของ inference path เดียวกับที่ backend/main.py ใช้ตอนรับเฟรมจากกล้อง
(hand detection ด้วย MediaPipe + feature extraction + RandomForest predict)

วิธีใช้:
    py -3.11 scripts/benchmark_inference.py [ระยะเวลาที่จับ (วินาที), ค่าเริ่มต้น 15]
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import cv2
import joblib
import mediapipe as mp
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from backend.features import landmarks_to_feature_vector  # noqa: E402

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
mp_hands = mp.solutions.hands


def main():
    duration_sec = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0

    bundle = joblib.load(MODEL_PATH)
    clf = bundle["model"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: เปิดกล้องไม่ได้")
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    latencies_ms = []
    hand_detected_count = 0

    try:
        # warm-up (ไม่นับ เพราะรอบแรกมักช้ากว่าปกติจาก lazy init ของ MediaPipe)
        for _ in range(5):
            cap.read()
            hands.process(np.zeros((480, 640, 3), dtype=np.uint8))

        end_time = time.perf_counter() + duration_sec
        while time.perf_counter() < end_time:
            success, frame = cap.read()
            if not success:
                continue

            t0 = time.perf_counter()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_detected_count += 1
                hand_landmarks = results.multi_hand_landmarks[0]
                vec = landmarks_to_feature_vector(hand_landmarks).reshape(1, -1)
                clf.predict_proba(vec)

            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
    finally:
        hands.close()
        cap.release()

    arr = np.array(latencies_ms)
    avg_ms = float(np.mean(arr))
    p95_ms = float(np.percentile(arr, 95))
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    print(f"เฟรมที่วัดได้: {len(arr)}")
    print(f"เฟรมที่เจอมือ: {hand_detected_count} ({hand_detected_count / len(arr) * 100:.1f}%)")
    print(f"Average latency: {avg_ms:.2f} ms/frame")
    print(f"P95 latency:     {p95_ms:.2f} ms/frame")
    print(f"Achievable FPS:  {fps:.1f}")


if __name__ == "__main__":
    main()
