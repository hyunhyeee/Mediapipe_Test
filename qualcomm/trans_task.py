# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # Task 파일 경로 설정
# tflite_model_path = "sign_language_model.tflite"
# task_file_path = "sign_language_model.task"

# # MediaPipe Model Maker로 task 모델 생성
# base_options = python.BaseOptions(model_asset_path=tflite_model_path)
# vision_options = vision.GestureRecognizerOptions(base_options=base_options)

# # Task 파일 저장 코드 (가상 메서드 사용)
# print(f"Task 파일이 {task_file_path}에 저장될 준비가 완료되었습니다.")


import os
import json
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

# 경로 설정
tflite_model_path = "sign_language_model.tflite"
task_file_path = os.path.join(os.getcwd(), "sign_language_model.task")

# BaseOptions 및 Vision Task 옵션 설정
base_options = python.BaseOptions(model_asset_path=tflite_model_path)
vision_options = {
    "model_asset_path": base_options.model_asset_path,
    "num_hands": 2,
    "score_threshold": 0.5
}

# Task 정보를 JSON 형태로 저장하기
task_data = {
    "task_name": "HandGestureRecognition",
    "model": tflite_model_path,
    "options": vision_options
}

# 파일 저장 로직
with open(task_file_path, 'w') as task_file:
    json.dump(task_data, task_file)

print(f"Task 파일이 '{task_file_path}'에 저장되었습니다.")
