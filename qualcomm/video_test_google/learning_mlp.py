import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve

# 데이터 불러오기
file_path = 'hand_landmarks_5.csv'
data = pd.read_csv(file_path)

# 데이터 확인
print("데이터 샘플:")
print(data.head())

# X, y 분리
X_data = data.drop(columns=['label']).values  # 특징 데이터
y_data = data['label'].values  # 라벨 데이터

# One-hot encoding for labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y_data.reshape(-1, 1))

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_data, y_encoded, test_size=0.2, random_state=42)

# MLP 모델 설계
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_data.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y_data)), activation='softmax')  # 클래스 개수만큼 출력
])

# 모델 컴파일
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train, y_train, epochs=30, batch_size=1, validation_split=0.2)

# 모델 저장
model.save('mlp_model_5.h5')

# 모델 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# 전체 예측 출력
predictions = model.predict(X_data)

# 예측된 라벨 출력
predicted_labels = [encoder.categories_[0][np.argmax(pred)] for pred in predictions]
print("예측된 라벨 분포:", {label: predicted_labels.count(label) for label in encoder.categories_[0]})

# 원본 데이터의 라벨 분포 출력
unique, counts = np.unique(y_data, return_counts=True)
print("원본 데이터 라벨 분포:", dict(zip(encoder.categories_[0], counts)))

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

# Confusion Matrix 계산
true_labels = np.argmax(y_encoded, axis=1)  # 원본 라벨 인덱스로 변환
predicted_indices = np.argmax(predictions, axis=1)
cm = confusion_matrix(true_labels, predicted_indices)

# Confusion Matrix 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.categories_[0], yticklabels=encoder.categories_[0])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Precision-Recall Curve 계산
plt.figure(figsize=(10, 8))

for i in range(len(encoder.categories_[0])):
    # 각 클래스에 대해 Precision, Recall, Threshold를 계산
    precision, recall, _ = precision_recall_curve(y_encoded[:, i], predictions[:, i])
    plt.plot(recall, precision, marker='.', label=f'Class {i} - {encoder.categories_[0][i]}')

plt.title('Precision-Recall Curve for All Classes')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.legend(loc='best')
plt.show()
