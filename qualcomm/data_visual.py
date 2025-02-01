import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve

# 데이터 로드 및 모델 로드
data = pd.read_csv('sign_language_korean.csv')
X = data.drop('label', axis=1)
y = pd.get_dummies(data['label'])

# 데이터 정규화
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 모델 로드
model = load_model('sign_language_model_40_5.h5')

# 예측
predictions = model.predict(X)
true_labels = np.argmax(y.values, axis=1)

# 한글 폰트 설정
matplotlib.rc('font', family='NanumGothic')  # 혹은 시스템에 설치된 한글 폰트 (예: 'Malgun Gothic' 등)
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 부호가 깨지지 않도록 설정


# 시각화 코드

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
