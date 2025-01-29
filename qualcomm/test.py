# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from sklearn.preprocessing import MinMaxScaler
# import pandas as pd
# import numpy as np
# import joblib # 추가

# # 데이터 로드
# data = pd.read_csv('sign_language_korean.csv')

# # 입력 데이터(X)와 출력 데이터(y) 분리
# X = data.drop('label', axis=1)
# y = pd.get_dummies(data['label'])  # 모든 label을 원핫 인코딩

# # 입력 데이터 정규화
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

# # 스케일러 저장 추가
# joblib.dump(scaler, 'scaler.pkl')

# # 모델 정의
# model = Sequential([
#     Dense(128, activation='relu', input_shape=(X.shape[1],)),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     Dropout(0.3),
#     Dense(len(y.columns), activation='softmax')  # 레이블 개수에 따라 출력층 조정
# ])

# # 모델 컴파일
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # 모델 학습
# model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# # 모델 저장 (파일 형식을 .keras로 변경)
# model.save('sign_language_model.h5')

# # 모델 평가
# test_loss, test_accuracy = model.evaluate(X, y)
# print(f"Test Loss: {test_loss}")
# print(f"Test Accuracy: {test_accuracy}")

# # 전체 예측 출력
# predictions = model.predict(X)  # X의 모든 샘플에 대해 예측

# # 예측된 라벨 출력 (각 샘플에 대해 예측된 라벨을 출력)
# predicted_labels = [y.columns[np.argmax(pred)] for pred in predictions]  # 각 샘플에 대해 예측된 라벨

# # 예측된 라벨들을 출력
# for i, label in enumerate(predicted_labels):
#     print(f"샘플 {i+1} 예측된 라벨: {label}")
    
# # 원본 데이터의 라벨 분포 출력
# print(data['label'].value_counts())




import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# 데이터 로드
data = pd.read_csv('sign_language_korean.csv')

# 입력 데이터(X)와 출력 데이터(y) 분리
X = data.drop('label', axis=1)
y = pd.get_dummies(data['label'])  # 모든 label을 원핫 인코딩

# 입력 데이터 정규화
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 스케일러 저장
joblib.dump(scaler, 'scaler.pkl')

# 모델 정의
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(y.columns), activation='softmax')  # 레이블 개수에 따라 출력층 조정
])

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(X, y, epochs=30, batch_size=1, validation_split=0.2)

# 모델 저장
model.save('sign_language_model.h5')

# 모델 평가
test_loss, test_accuracy = model.evaluate(X, y)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# 전체 예측 출력
predictions = model.predict(X)  # X의 모든 샘플에 대해 예측

# 예측된 라벨 출력 (각 샘플에 대해 예측된 라벨을 출력)
predicted_labels = [y.columns[np.argmax(pred)] for pred in predictions]  # 각 샘플에 대해 예측된 라벨

# # 예측된 라벨들을 출력
# for i, label in enumerate(predicted_labels):
#     print(f"샘플 {i+1} 예측된 라벨: {label}")
    
# 원본 데이터의 라벨 분포 출력
print(data['label'].value_counts())

# 한글 폰트 설정
matplotlib.rc('font', family='NanumGothic')  # 혹은 시스템에 설치된 한글 폰트 (예: 'Malgun Gothic' 등)
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 부호가 깨지지 않도록 설정


# 데이터 시각화
# 1. 학습 및 검증 정확도 시각화
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# 2. 학습 및 검증 손실 시각화
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# 3. 데이터의 라벨 분포 시각화
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=data)
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# True labels (convert to array for comparison)
true_labels = np.argmax(y.values, axis=1)  # Convert one-hot encoded labels to integers

# Confusion Matrix 계산
cm = confusion_matrix(true_labels, np.argmax(predictions, axis=1))  # Use argmax to get predicted labels

# Confusion Matrix 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=y.columns, yticklabels=y.columns)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# Precision-Recall Curve 계산
from sklearn.metrics import precision_recall_curve

# 다중 클래스 Precision-Recall Curve 계산
plt.figure(figsize=(10, 8))  # 하나의 그래프 크기 설정

for i in range(len(y.columns)):
    # 각 클래스에 대해 Precision, Recall, Threshold를 계산
    precision, recall, _ = precision_recall_curve(y.values[:, i], predictions[:, i])
    
    # Precision-Recall Curve를 같은 그래프에 그리기
    plt.plot(recall, precision, marker='.', label=f'Class {i} - {y.columns[i]}')

# 그래프 제목, 레이블 추가
plt.title('Precision-Recall Curve for All Classes')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.legend(loc='best')  # 범례를 그래프의 최적 위치에 추가
plt.show()
