import os
import csv
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from PIL import Image as PILImage, ImageDraw, ImageFont

# 제스처 매핑 (영어 입력값과 한글 지화 매핑)
gesture = {
    0: 'ga', 1: 'na', 2: 'da', 3: 'ra', 4: 'ma', 5: 'ba', 6: 'sa', 7: 'a', 8: 'ja', 9: 'cha',
    10: 'ka', 11: 'ta', 12: 'pa', 13: 'ha', 14: 'aa', 15: 'ya', 16: 'eo', 17: 'yeo', 18: 'o', 
    19: 'yo', 20: 'u', 21: 'yu', 22: 'eu', 23: 'lee', 24: 'ae', 25: 'e', 26: 'oe', 27: 'wi', 
    28: 'yae', 29: 'ye', 30: 'ui'
}

hangeul_gesture = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅊ',
    10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ', 14: 'ㅏ', 15: 'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ', 
    19: 'ㅛ', 20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ', 23: 'ㅣ', 24: 'ㅐ', 25: 'ㅔ', 26: 'ㅚ', 27: 'ㅟ', 
    28: 'ㅒ', 29: 'ㅖ', 30: 'ㅢ'
}

# CSV 파일 경로
csv_path = 'sign_language_korean.csv'

# CSV 파일이 비어 있다면 헤더 추가
if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)])

# CSV 파일 열기 (append 모드)
data_file = open(csv_path, mode='a', newline='', encoding='utf-8')
csv_writer = csv.writer(data_file)

# Mediapipe 손 인식 (Hands 객체 사용)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hands 객체 생성 (손 감지 모드 설정)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 카메라 실행
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 카메라 실행 중 손 랜드마크 감지
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패.")
        break

    # BGR 이미지를 RGB로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 손 랜드마크 예측
    results = hands.process(frame_rgb)

    # 손 랜드마크 처리
    row = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 각 손의 랜드마크는 hand_landmarks 객체 내에 'landmark' 속성이 있습니다.
            if hasattr(hand_landmarks, 'landmark'):  # hand_landmarks가 LandmarkList 형식인지 확인
                for lm in hand_landmarks.landmark:  # hand_landmarks는 이미 LandmarkList 형식입니다.
                    row.extend([lm.x, lm.y, lm.z])

                # 손 랜드마크 시각화
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2), 
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))
    else:
        print("손을 인식하지 못했습니다.")  # 손을 인식하지 못한 경우 메시지 출력

    # 한글 텍스트 출력
    pil_img = PILImage.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    # 폰트 경로 확인 후 변경 (Windows 환경에서 한글 폰트 경로)
    font_path = 'C:/Windows/Fonts/malgun.ttf'
    font = ImageFont.truetype(font_path, 30)

    draw.text((50, 50), '입력 후 엔터를 누르세요!', font=font, fill=(255, 255, 0))
    frame = np.array(pil_img)

    # 화면에 출력
    cv2.imshow('Mediapipe Hand Detection', frame)

    # 키보드 입력 처리
    key = cv2.waitKey(1) & 0xFF  # 키 입력 대기
    if key == ord('q'):  # q를 누르면 종료
        print("프로그램 종료")
        break

    if key == 13:  # 엔터 키가 눌렸을 때
        user_input = input("영어 제스처 입력 (예: ga): ")
        # 입력 값이 gesture 매핑에 있는 경우
        if user_input in gesture.values():
            index = list(gesture.values()).index(user_input)
            hangeul_char = hangeul_gesture[index]
            print(f"입력된 제스처: {user_input} -> 한글: {hangeul_char}")

            # CSV 파일에 저장
            csv_writer.writerow([hangeul_char] + row)
            print(f"데이터 저장 완료: {hangeul_char}")
        else:
            print(f"'{user_input}'은 유효하지 않은 입력입니다. 다시 입력하세요.")

data_file.close()
cap.release()
cv2.destroyAllWindows()
