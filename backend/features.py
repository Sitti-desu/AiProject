import mediapipe as mp
import numpy as np

# ตั้งค่าให้รองรับ 2 มือ
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# เพิ่ม feature_names กลับเข้าไป (สำหรับมือซ้าย 63 ค่า + มือขวา 63 ค่า = 126 ค่า)
feature_names = []
for hand in ['left', 'right']:
    for i in range(21):
        feature_names.extend([f'{hand}_x{i}', f'{hand}_y{i}', f'{hand}_z{i}'])

def landmarks_to_feature_vector(results):
    # สร้าง Array เติมเลข 0 เตรียมไว้
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
                
            if label == 'Left':
                left_hand = np.array(features)
            elif label == 'Right':
                right_hand = np.array(features)

    # นำข้อมูล 2 มือมาต่อกัน
    return np.concatenate([left_hand, right_hand]).tolist()