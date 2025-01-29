# import tensorflow as tf

# # 모델 로드
# model = tf.keras.models.load_model('sign_language_model.task')

# # TensorFlow Lite 변환기 초기화
# converter = tf.lite.TFLiteConverter.from_keras_model(model)

# # 입력/출력 타입 설정 (float32 유지)
# converter.experimental_new_converter = True
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# # 변환
# tflite_model = converter.convert()

# # 모델 저장
# with open('sign_language_model.tflite', 'wb') as f:
#     f.write(tflite_model)


# # TensorFlow Lite 모델 로드 및 테스트
# interpreter = tf.lite.Interpreter(model_path='sign_language_model.tflite')
# interpreter.allocate_tensors()

# # 입력 및 출력 세부 정보 확인
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# print("입력 정보:", input_details)
# print("출력 정보:", output_details)


import tensorflow as tf

# 기존 모델 로드
model = tf.keras.models.load_model('sign_language_model.h5')

# TensorFlow Lite 모델로 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# TFLite 모델 저장
with open('sign_language_model.tflite', 'wb') as f:
    f.write(tflite_model)
