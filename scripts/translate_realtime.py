"""
translate_realtime.py
เปิดกล้อง ทำนายท่ามือแบบเรียลไทม์ แล้วขึ้นซับ (subtitle) ด้านล่างจอ

วิธีใช้:
    python translate_realtime.py

กด 'q' เพื่อออก, 'c' เพื่อล้างประโยคที่สะสมไว้
"""

import os
import sys
from collections import deque, Counter

import cv2
import joblib
import mediapipe as mp
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from backend.features import landmarks_to_feature_vector  # noqa: E402

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

SMOOTHING_WINDOW = 10
STABLE_RATIO = 0.7
CONF_THRESHOLD = 0.55
IGNORE_LABEL = "no_sign"
MAX_SENTENCE_WORDS = 8


def draw_subtitle_bar(image, text):
    h, w = image.shape[:2]
    bar_height = 60
    overlay = image.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    if text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = max((w - text_w) // 2, 10)
        y = h - bar_height // 2 + text_h // 2
        cv2.putText(image, text, (x, y), font, font_scale, (255, 255, 255), thickness)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ไม่พบโมเดลที่ {MODEL_PATH} — รัน train_model.py ก่อน")
        return

    bundle = joblib.load(MODEL_PATH)
    clf = bundle["model"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: เปิดกล้องไม่ได้")
        return

    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
    sentence_words = deque(maxlen=MAX_SENTENCE_WORDS)
    last_committed_label = None
    current_display_label = ""

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        try:
            while True:
                success, image = cap.read()
                if not success:
                    continue

                image = cv2.flip(image, 1)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                predicted_label = None
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    vec = landmarks_to_feature_vector(hand_landmarks).reshape(1, -1)
                    probs = clf.predict_proba(vec)[0]
                    best_idx = int(np.argmax(probs))
                    best_label = clf.classes_[best_idx]
                    best_conf = probs[best_idx]

                    if best_conf >= CONF_THRESHOLD:
                        predicted_label = best_label

                recent_predictions.append(predicted_label)

                valid = [p for p in recent_predictions if p is not None]
                stable_label = None
                if len(valid) >= max(3, int(SMOOTHING_WINDOW * 0.5)):
                    label, cnt = Counter(valid).most_common(1)[0]
                    if cnt / len(recent_predictions) >= STABLE_RATIO:
                        stable_label = label

                if stable_label is not None:
                    current_display_label = stable_label
                    if stable_label != IGNORE_LABEL and stable_label != last_committed_label:
                        sentence_words.append(stable_label)
                        last_committed_label = stable_label
                elif not valid:
                    last_committed_label = None
                    current_display_label = ""

                cv2.putText(
                    image, f"Now: {current_display_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )

                sentence_text = " ".join(sentence_words)
                draw_subtitle_bar(image, sentence_text)

                cv2.imshow("Sign Language Translator", image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord("c"):
                    sentence_words.clear()
                    last_committed_label = None
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()