import joblib
import cv2
import mediapipe as mp
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

# 모델 및 스케일러 로드
model = load_model('sign_language_model_40_1.h5')
scaler = joblib.load('scaler_40_1.pkl')

# Mediapipe 초기화3
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hangeul_gesture = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅊ',
    10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ', 14: 'ㅏ', 15: 'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ', 
    19: 'ㅛ', 20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ', 23: 'ㅣ', 24: 'ㅐ', 25: 'ㅔ', 26: 'ㅚ', 27: 'ㅟ', 
    28: 'ㅒ', 29: 'ㅖ', 30: 'ㅢ'
}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                hand_landmarks_list = []

                # 랜드마크 정보 수집
                for lm in landmarks:
                    hand_landmarks_list.extend([lm.x, lm.y, lm.z])

                # 스케일러 변환 (데이터프레임으로 변환)
                columns = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)]
                hand_landmarks_df = pd.DataFrame([hand_landmarks_list], columns=columns)
                hand_landmarks_scaled = scaler.transform(hand_landmarks_df)

                # 모델 예측
                prediction = model.predict(hand_landmarks_scaled)
                predicted_label = np.argmax(prediction)
                prediction_confidence = np.max(prediction)

                # 예측 결과 처리
                if prediction_confidence < 0.7:
                    hangeul_char = "학습되지 않은 동작입니다."
                else:
                    hangeul_char = hangeul_gesture.get(predicted_label, "학습되지 않은 동작입니다.")

                # 결과 화면에 한글 텍스트 출력
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                font = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 50)
                draw.text((50, 50), hangeul_char, font=font, fill=(0, 0, 0))  # 글자 색을 검은색으로 설정
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                # 랜드마크 그리기
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Mediapipe Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()