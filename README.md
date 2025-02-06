# Mediapipe_Test

## 실행방식
1. 가상환경 실행 venv\Scripts\activate -> 없으면 라이브러리 설치치
2. cd qualcomm 디렉터리 이동

동작원리
1. ```python mediapipe_test.py```

-> 미디어파이프를 실행시켜 손동작 좌표를 저장시키는 코드
  - 카메라가 켜지면 카메라 화면을 누르고 원하는 동작을 한 후 엔터키를 누름
  - 터미널에 해당 동작의 영어를 적음 (ex. ㄱ이면 ga 입력)
  - 종료를 원하면 카메라 화면을 누르며 q 입력

2. CSV파일에 좌표가 저장된 것을 확인

3. ```python test.py```

-> 좌표로 저장된 csv파일을 학습시키는 코드
  이때 데이터 시각화가 뜸
  - 학습이 완료되면 sign_language_model.h5 생성됨

4. ```python mediapipe_result.py```

-> 학습된 모델을 불러와 카메라에서 확인해보는 코드
  - 학습한 동작을 실행하면 카메라 화면에 해당 글자가 뜸뜸


</br><hr>
- jihwa_test - 지화 데이터셋으로 구성 (이미지 좌표)
 - video_test_google - 구글 미디어파이프를 활용하여 영상 데이터셋으로 구성
