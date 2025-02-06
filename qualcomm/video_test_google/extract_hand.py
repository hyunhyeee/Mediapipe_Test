import mediapipe as mp
import cv2
import numpy as np
import os
import pandas as pd

# MediaPipe 초기 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_hand_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 전처리
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # 양손 랜드마크 추출
        hand_landmarks_data = [None, None]  # [왼손, 오른손] 초기화

        if results.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])  # x, y, z 좌표 추가
                
                # 오른손이 먼저 감지될 수 있으므로 손 위치에 맞게 저장
                handedness = results.multi_handedness[hand_index].classification[0].label
                if handedness == 'Left':
                    hand_landmarks_data[0] = landmarks
                else:
                    hand_landmarks_data[1] = landmarks

            # 손이 하나만 감지된 경우에도 데이터 처리
            if hand_landmarks_data[0] is None:
                hand_landmarks_data[0] = [0] * 63  # 21개의 랜드마크 * 3 (x, y, z)
            if hand_landmarks_data[1] is None:
                hand_landmarks_data[1] = [0] * 63  # 21개의 랜드마크 * 3 (x, y, z)

            # 두 손 랜드마크 데이터 결합
            combined_landmarks = hand_landmarks_data[0] + hand_landmarks_data[1]
            landmarks_list.append(combined_landmarks)

    cap.release()
    return np.array(landmarks_list)

# 데이터 수집 경로 설정
base_data_path = './videos/'
label_folders = ['love', 'hello', 'thanks', 'sorry', 'itsok']
data = []

for folder_name in label_folders:
    folder_path = os.path.join(base_data_path, folder_name)
    if not os.path.exists(folder_path):
        print(f"폴더 {folder_path}가 존재하지 않습니다.")
        continue

    # 각 폴더 내 영상 파일 읽기
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.avi')]
    print(f"{folder_name} 폴더 내 {len(video_files)}개의 영상 처리 중...")

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        print(f"Processing {video_path}...")
        landmarks = extract_hand_landmarks(video_path)
        
        # 데이터 수집: 각 프레임마다 라벨 포함하여 저장
        if landmarks.size > 0:
            for frame_landmarks in landmarks:
                data.append(list(frame_landmarks) + [folder_name])

# 데이터를 DataFrame으로 변환 및 저장
columns = [f"x{i//3}_y{i//3}_z{i//3}" for i in range(126)] + ["label"]
df = pd.DataFrame(data, columns=columns)
df.to_csv('hand_landmarks_5.csv', index=False)
print("hand_landmarks_5.csv 저장 완료!")




# import mediapipe as mp
# import cv2
# import numpy as np
# import os
# import pandas as pd

# # MediaPipe 초기 설정
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils

# def extract_hand_landmarks(video_path):
#     cap = cv2.VideoCapture(video_path)
#     landmarks_list = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # 이미지 전처리
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(frame_rgb)

#         # 양손 랜드마크 추출
#         hand_landmarks_data = [[0] * 63, [0] * 63]  # [왼손, 오른손] 초기화

#         if results.multi_hand_landmarks:
#             for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
#                 landmarks = []
#                 for lm in hand_landmarks.landmark:
#                     landmarks.extend([lm.x, lm.y, lm.z])  # x, y, z 좌표 추가

#                 # 오른손과 왼손 분류
#                 handedness = results.multi_handedness[hand_index].classification[0].label
#                 if handedness == 'Left':
#                     hand_landmarks_data[0] = landmarks
#                 else:
#                     hand_landmarks_data[1] = landmarks

#         # 두 손 랜드마크 데이터 결합
#         combined_landmarks = hand_landmarks_data[0] + hand_landmarks_data[1]
#         landmarks_list.append(combined_landmarks)

#     cap.release()
#     return np.array(landmarks_list)

# # 데이터 수집 경로 설정
# base_data_path = './videos/'
# label_folders = ['love', 'hello', 'thanks', 'sorry', 'itsok']
# data = []

# for folder_name in label_folders:
#     folder_path = os.path.join(base_data_path, folder_name)
#     if not os.path.exists(folder_path):
#         print(f"폴더 {folder_path}가 존재하지 않습니다.")
#         continue

#     # 각 폴더 내 영상 파일 읽기
#     video_files = [f for f in os.listdir(folder_path) if f.endswith('.avi')]
#     print(f"{folder_name} 폴더 내 {len(video_files)}개의 영상 처리 중...")

#     for video_file in video_files:
#         video_path = os.path.join(folder_path, video_file)
#         print(f"Processing {video_path}...")
#         try:
#             landmarks = extract_hand_landmarks(video_path)

#             # 데이터 수집: 각 프레임마다 라벨 포함하여 저장
#             if landmarks.size > 0:
#                 for frame_landmarks in landmarks:
#                     data.append(list(frame_landmarks) + [folder_name])
#         except Exception as e:
#             print(f"오류 발생: {video_path} 처리 중 {str(e)}")

# # 데이터를 DataFrame으로 변환 및 저장
# columns = [f"x{i//3}_y{i//3}_z{i//3}" for i in range(126)] + ["label"]
# df = pd.DataFrame(data, columns=columns)
# df.to_csv('hand_landmarks2.csv', index=False)
# print("hand_landmarks2.csv 저장 완료!")






# import mediapipe as mp
# import cv2
# import numpy as np
# import os
# import pandas as pd

# # MediaPipe 초기 설정
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils

# def extract_hand_landmarks(video_path):
#     cap = cv2.VideoCapture(video_path)
#     landmarks_list = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # 이미지 전처리
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(frame_rgb)

#         # 양손 랜드마크 추출
#         hand_landmarks_data = [[0] * 63, [0] * 63]  # [왼손, 오른손] 초기화

#         if results.multi_hand_landmarks:
#             for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
#                 landmarks = []
#                 for lm in hand_landmarks.landmark:
#                     landmarks.extend([lm.x, lm.y, lm.z])  # x, y, z 좌표 추가

#                 # 오른손과 왼손 분류
#                 handedness = results.multi_handedness[hand_index].classification[0].label
#                 if handedness == 'Left':
#                     hand_landmarks_data[0] = landmarks
#                 else:
#                     hand_landmarks_data[1] = landmarks

#         # 두 손 랜드마크 데이터 결합
#         combined_landmarks = hand_landmarks_data[0] + hand_landmarks_data[1]
#         landmarks_list.append(combined_landmarks)

#     cap.release()

#     # 영상 하나의 단위로 랜드마크 데이터를 하나로 결합
#     if landmarks_list:
#         # 영상 전체의 랜드마크 평균값을 계산하여 하나의 벡터로 만듦
#         average_landmarks = np.mean(landmarks_list, axis=0)
#         return average_landmarks
#     return None

# # 데이터 수집 경로 설정
# base_data_path = './videos/'
# label_folders = ['love', 'hello', 'thanks', 'sorry', 'itsok']
# data = []

# for folder_name in label_folders:
#     folder_path = os.path.join(base_data_path, folder_name)
#     if not os.path.exists(folder_path):
#         print(f"폴더 {folder_path}가 존재하지 않습니다.")
#         continue

#     # 각 폴더 내 영상 파일 읽기
#     video_files = [f for f in os.listdir(folder_path) if f.endswith('.avi')]
#     print(f"{folder_name} 폴더 내 {len(video_files)}개의 영상 처리 중...")

#     for video_file in video_files:
#         video_path = os.path.join(folder_path, video_file)
#         print(f"Processing {video_path}...")
#         try:
#             landmarks = extract_hand_landmarks(video_path)

#             # 데이터 수집: 한 영상에 대해 평균 랜드마크 벡터와 라벨을 함께 저장
#             if landmarks is not None:
#                 data.append(list(landmarks) + [folder_name])
#         except Exception as e:
#             print(f"오류 발생: {video_path} 처리 중 {str(e)}")

# # 데이터를 DataFrame으로 변환 및 저장
# columns = [f"x{i//3}_y{i//3}_z{i//3}" for i in range(126)] + ["label"]
# df = pd.DataFrame(data, columns=columns)
# df.to_csv('hand_landmarks_4.csv', index=False)
# print("hand_landmarks_4.csv 저장 완료!")
