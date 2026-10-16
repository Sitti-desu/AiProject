import csv
import os
import sys
import time  # เพิ่มไลบรารี time เข้ามา

import cv2
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from backend.features import feature_names, landmarks_to_feature_vector  # noqa: E402

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "gestures.csv")

def ensure_csv_header():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["label"] + feature_names())

def append_sample(label, feature_vector):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([label] + list(feature_vector))

def collect_for_label(cap, hands, label):
    count = 0
    print(f"\n=== กำลังเก็บท่า: '{label}' ===")
    print("กด SPACE เพื่อเริ่มนับถอยหลัง | 'n' ไปท่าถัดไป | 'q' บันทึกและออก")

    # ตัวแปรสำหรับจัดการเวลา
    is_countdown = False
    is_recording = False
    countdown_start = 0
    record_start = 0

    while True:
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        display = image.copy()
        
        # วาดเส้นบนมือ (รองรับ 2 มือ)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        current_time = time.time()

        # จัดการสถานะการนับถอยหลัง (3 วินาที)
        if is_countdown:
            elapsed = current_time - countdown_start
            remain = 3 - int(elapsed)
            if remain > 0:
                cv2.putText(display, f"Starting in: {remain}", (display.shape[1]//2 - 150, display.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            else:
                is_countdown = False
                is_recording = True
                record_start = time.time()

        # จัดการสถานะการบันทึก (10 วินาที)
        if is_recording:
            elapsed = current_time - record_start
            remain = 10 - int(elapsed)
            if remain > 0:
                cv2.putText(display, f"Recording: {remain}s", (display.shape[1]//2 - 150, display.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                
                # บันทึกข้อมูลอัตโนมัติรัวๆ
                if results.multi_hand_landmarks is not None:
                    vec = landmarks_to_feature_vector(results)
                    append_sample(label, vec)
                    count += 1
            else:
                is_recording = False

        # ข้อความ UI ปกติ
        cv2.rectangle(display, (0, 0), (display.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(
            display, f"Label: {label} | Samples: {count}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if is_recording else (255, 255, 255), 2,
        )
        cv2.putText(
            display, "SPACE=Start Record  n=next label  q=save&quit", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        cv2.imshow("Collect Sign Data", display)
        key = cv2.waitKey(1) & 0xFF

        # ปุ่มกด
        if key == ord(" "):
            if not is_countdown and not is_recording:
                is_countdown = True
                countdown_start = time.time()
        elif key == ord("n"):
            return "next"
        elif key == ord("q"):
            return "quit"

def main():
    ensure_csv_header()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: เปิดกล้องไม่ได้")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        try:
            while True:
                label = input(
                    "\nพิมพ์ชื่อ label (เว้นว่างแล้ว Enter เพื่อออก): "
                ).strip()
                if not label:
                    break

                result = collect_for_label(cap, hands, label)
                if result == "quit":
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    print(f"\nบันทึกข้อมูลไว้ที่: {CSV_PATH}")

if __name__ == "__main__":
    main()