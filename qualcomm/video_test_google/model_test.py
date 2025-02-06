# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image, ImageDraw, ImageFont
# import time

# # 모델 로드
# model = load_model('mlp_model.h5')

# # Mediapipe 초기화
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils

# # 수어 단어 라벨
# hangeul_gesture = {
#     0: "love",
#     1: "hello",
#     2: "thanks",
#     3: "sorry",
#     4: "itsok"
# }

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("카메라를 열 수 없습니다.")
#     exit()

# hand_state = "inactive"
# start_time = None
# gesture_threshold = 1.5  # 1.5초 이상 지속되어야 동작으로 인정
# hangeul_char = ""
# gesture_recognized = False  # 단어가 인식된 상태를 추적하는 변수

# with mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=2,  # 양손 추적
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# ) as hands:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(frame_rgb)

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 landmarks = hand_landmarks.landmark
#                 hand_landmarks_list = []

#                 # 랜드마크 정보 수집
#                 for lm in landmarks:
#                     hand_landmarks_list.extend([lm.x, lm.y, lm.z])

#                 # 모델 입력 차원에 맞게 126 차원으로 맞추기 (예시: zero-padding)
#                 hand_landmarks_array = np.array(hand_landmarks_list)
                
#                 if hand_landmarks_array.shape[0] < 126:
#                     # 부족한 부분을 0으로 채움
#                     hand_landmarks_array = np.pad(hand_landmarks_array, (0, 126 - hand_landmarks_array.shape[0]), 'constant', constant_values=0)
                
#                 hand_landmarks_array = hand_landmarks_array.reshape(1, -1)  # 1차원 배열로 변환하여 모델에 입력

#                 # 모델 예측
#                 prediction = model.predict(hand_landmarks_array)
#                 predicted_label = np.argmax(prediction)
#                 prediction_confidence = np.max(prediction)

#                 # 예측 결과 처리
#                 if prediction_confidence < 0.7:
#                     hangeul_char = "학습되지 않은 동작입니다."
#                     hand_state = "inactive"  # 확신도가 낮으면 동작을 인식하지 않음
#                 else:
#                     hangeul_char = hangeul_gesture.get(predicted_label, "학습되지 않은 동작입니다.")
#                     if hand_state == "inactive" and not gesture_recognized:
#                         hand_state = "active"
#                         start_time = time.time()  # 손동작 시작 시간 기록

#                 # 손동작이 1.5초 이상 지속되었으면 결과를 출력
#                 if hand_state == "active" and time.time() - start_time > gesture_threshold:
#                     gesture_recognized = True  # 단어가 인식되었음을 표시
#                     hand_state = "finished"  # 손동작 완료 상태

#                 # 예측 결과 디버깅 출력
#                 print(f"예측된 라벨: {predicted_label}, 확신도: {prediction_confidence:.2f}")

#                 # 단어 출력: 단어가 인식되고 화면에 계속 유지되도록 함
#                 if gesture_recognized:
#                     # 결과 화면에 한글 텍스트 출력
#                     pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                     draw = ImageDraw.Draw(pil_img)
#                     font = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 50)

#                     # 텍스트를 새로 그리기
#                     draw.text((50, 50), hangeul_char, font=font, fill=(0, 0, 0))  # 글자 색을 검은색으로 설정
#                     frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

#                     # 랜드마크 그리기
#                     mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                     # 상태 초기화 (새로운 동작을 기다림)
#                     if time.time() - start_time > 3:  # 3초 동안 단어가 계속 출력되도록 설정
#                         gesture_recognized = False  # 3초가 지나면 단어를 초기화
#                         hand_state = "inactive"  # 새로운 동작을 기다림
#                         hangeul_char = ""  # 출력된 텍스트 초기화

#         # 결과 화면에 출력
#         cv2.imshow("Hand Gesture Recognition", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()





import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import collections
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# 모델 로드
model = load_model('mlp_model_3.h5')

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 수어 단어 라벨
hangeul_gesture = {
    0: "love",
    1: "hello",
    2: "thanks",
    3: "sorry",
    4: "itsok"
}

# Prediction 관련 변수
prediction_window = 10  # 최근 10개 프레임 사용
prediction_queue = collections.deque(maxlen=prediction_window)
gesture_threshold = 1.5  # 1.5초 이상 지속되어야 동작으로 인정
hand_state = "inactive"
start_time = None
gesture_recognized = False  # 단어가 인식된 상태를 추적하는 변수
hangeul_char = ""

# 카메라 캡처 설정
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
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
                hand_landmarks_list = [lm.x for lm in landmarks] + \
                                       [lm.y for lm in landmarks] + \
                                       [lm.z for lm in landmarks]

                hand_landmarks_array = np.array(hand_landmarks_list)

                # Padding for input shape consistency
                if hand_landmarks_array.shape[0] < 126:
                    hand_landmarks_array = np.pad(hand_landmarks_array, 
                                                  (0, 126 - hand_landmarks_array.shape[0]), 
                                                  'constant')

                hand_landmarks_array = hand_landmarks_array.reshape(1, -1)
                prediction = model.predict(hand_landmarks_array)
                predicted_label = np.argmax(prediction)

                # 예측 결과를 prediction_queue에 저장
                prediction_queue.append(predicted_label)

                # 가장 빈도가 높은 라벨 찾기
                if len(prediction_queue) == prediction_window:
                    most_common_label, count = collections.Counter(prediction_queue).most_common(1)[0]

                    # 충분히 빈도 높은 경우 해당 단어 확정
                    if count > prediction_window // 2:
                        hangeul_char = hangeul_gesture.get(most_common_label, "학습되지 않은 동작입니다.")
                        if hand_state == "inactive":
                            hand_state = "active"
                            start_time = time.time()  # 손동작 시작 시간 기록

                    # 손동작이 1.5초 이상 지속되었으면 결과를 출력
                    if hand_state == "active" and time.time() - start_time > gesture_threshold:
                        gesture_recognized = True
                        hand_state = "finished"  # 손동작 완료 상태

                # 단어 출력: 단어가 인식되고 화면에 계속 유지되도록 함
                if gesture_recognized:
                    # 결과 화면에 한글 텍스트 출력
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                    font = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 50)

                    # 텍스트를 새로 그리기
                    draw.text((50, 50), hangeul_char, font=font, fill=(0, 0, 0))  # 글자 색을 검은색으로 설정
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                    # 랜드마크 그리기
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # 상태 초기화 (새로운 동작을 기다림)
                    if time.time() - start_time > 3:  # 3초 동안 단어가 계속 출력되도록 설정
                        gesture_recognized = False  # 3초가 지나면 단어를 초기화
                        hand_state = "inactive"  # 새로운 동작을 기다림
                        hangeul_char = ""  # 출력된 텍스트 초기화

        # 결과 화면에 출력
        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
