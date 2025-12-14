#  AI 기반 생활 폐기물 분류 시스템


## 1. 프로젝트 개요 및 목표

| 항목 | 내용 |
| :--- | :--- |
| **프로젝트명** | EfficientNet 기반 실시간 생활 폐기물 분류 시스템 |
| **프로젝트 목표** | 5가지 핵심 폐기물 카테고리(캔, 유리, 종이, 플라스틱, 일반쓰레기)를 정확히 식별하는 AI 모델을 구축하고 웹 서비스에 배포하여 재활용 효율 증대 |
| **개발 동기/배경** | 복합 재질 및 오염된 폐기물의 모호한 분류로 인한 재활용 효율 저하 문제 해결 및 오분류 방지 |

---

### 1.1 주요 목표 요약

1.  **정밀 자동 분류:** 5가지 핵심 카테고리를 정확하게 식별하는 AI 모델 구축.
2.  **웹 서비스:** 사용자가 웹캠이나 이미지 업로드를 통해 손쉽게 접근 가능한 **반응형 웹 인터페이스** 구현.
3.  **오분류 개선:** 기존 모델들이 취약했던 **'흰색 배경 편향'** 및 **'일반쓰레기 혼동'** 문제를 데이터 및 전처리 기술로 해결.

   
---



##  2. 개발 환경 및 이론적 배경

### 2.1 개발 환경 (Environment)

| 항목 | 요소 | 내용 |
| :--- | :--- | :--- |
| **AI 모델 프레임워크** | PyTorch (TorchVision) | 모델 개발 및 학습 |
| **모델 아키텍처** | **EfficientNet-B0** | 경량화와 고성능을 동시에 달성 |
| **백엔드 서버** | **FastAPI** (Uvicorn) | 비동기 추론 서버 구축 |
| **프론트엔드** | HTML5, JavaScript (WebRTC), Tailwind CSS | 반응형 웹 UI 및 카메라 제어 |
| **GPU 가속** | CUDA | 고속 모델 학습 및 추론 |

### 2.2 모델 아키텍처 및 학습 전략

* **EfficientNet-B0 선정:** 모델의 **깊이(Depth), 너비(Width), 해상도(Resolution)**를 최적으로 결합하는 **Compound Scaling** 원리가 적용된 모델입니다.  웹 서비스 환경에 적합하도록 경량화와 고성능을 동시에 달성할 수 있습니다.
* **전이 학습 (Transfer Learning):** ImageNet으로 사전 학습된 가중치(Pre-trained Weights)를 활용하여, 제한된 커스텀 데이터셋으로도 빠른 수렴 속도와 높은 정확도를 확보하였습니다.

---

##  3. 시스템 설계 및 처리 프로세스

### 3.1 시스템 구성 요소 및 역할

| 구성 요소 | 역할 |
| :--- | :--- |
| **클라이언트 (웹 UI)** | 사용자 이미지 입력 (웹캠/업로드) 및 결과 시각화 |
| **이미지 전처리** | 입력 이미지를 224x224로 리사이즈 및 정규화 |
| **CNN 분류 모델** | EfficientNet-B0이 이미지 특징을 추출하고 5가지 클래스로 분류 |
| **FastAPI 서버** | 클라이언트 요청 수신, AI 추론 실행, JSON 형태로 예측 결과 반환 |

### 3.2 데이터셋 구성

* **기본 5개 클래스:** Can, General, Glass, Paper, Plastic
* **Can 클래스 확장:** 단순 음료 캔뿐만 아니라 알루미늄 트레이, 구겨진 포일 등 **재활용 가능한 금속류(Metal)** 전반을 인식하도록 데이터 범위를 확장하였습니다.

---

## 4. 성능 평가 및 결과 분석

### 4.1 모델 최종 성능

| 항목 | 성능 지표 | 값 |
| :--- | :--- | :--- |
| **총 학습량 (Epochs)** | Iterations | **20 Epochs** |
| **최고 검증 정확도 (Best Validation Acc)** | Accuracy | **[95.88]%** |
| **GPU (CUDA) 추론 시간** | 평균 | 약 0.2초 ~ 0.5초 (실시간 처리 가능) |
| **CPU 추론 시간** | 평균 | 약 1.5초 ~ 2.5초 |

### 4.2 결과 분석 및 결론

최종 모델은 데이터 보강과 증강(Augmentation) 강화를 통해, 프로젝트의 가장 큰 난제였던 **'흰색 배경에서의 일반쓰레기 오분류' 문제**를 성공적으로 해결하며 **[95.88]%**의 검증 정확도를 달성했습니다. 또한, 프론트엔드 장치 제어 로직 개선을 통해 다양한 사용자 환경에서의 호환성을 확보하였습니다.


---

##  5. 핵심 문제 해결 (Problem Solving) 

본 프로젝트의 가장 중요한 성과는 실제 서비스 환경에서 발생하는 **치명적인 오분류 및 하드웨어 충돌 문제**를 공학적으로 해결한 과정에 있습니다.

### 5.1 일반쓰레기 오분류 및 데이터 불균형 해결

* **문제:** 초기 모델이 오염된 일반쓰레기를 깨끗한 플라스틱으로 오인하는 경향 발생 (General 클래스 인식률 저하).
* **해결 (Hard Negative Mining):** 오염되거나 복합 재질인 실제 일반쓰레기 이미지 101장을 직접 수집하여(Hard Negative Mining) `dataset/general` 폴더를 보강했습니다.
* **결과:** 재학습 후 General 클래스 인식률이 대폭 향상되어 플라스틱과의 혼동이 최소화되었습니다.

### 5.2 [Critical] 흰색 배경 과적합(Background Bias Overfitting) 해결

* **문제:** 일반쓰레기를 **흰색 배경(책상, 벽)**에서 촬영 시, 모델이 객체의 특징이 아닌 **'흰색 배경'이라는 편향된 정보**를 학습하여 '플라스틱'으로 오분류하는 심각한 배경색 과적합 문제가 발생했습니다.
* **해결 (데이터 증강 강화 - 핵심 전략):** `trainer.py`의 전처리 로직 중 **`ColorJitter` 파라미터를 강화 적용**했습니다.
    * **적용 코드:** `transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)`
    * **목적:** 조명 및 색상 환경을 강하게 변조하여 모델이 배경색에 덜 민감하고 **물체 고유의 형상 및 질감**에 집중하도록 강제하여 일반화 성능을 높였습니다.

### 5.3 실시간 카메라 장치 충돌 해결 (Frontend)

* **문제:** 사용자 노트북에 **IR 카메라**나 **가상 카메라** 등 여러 장치가 충돌하여 `getUserMedia()`가 적절한 장치 스트림을 가져오는 데 실패하는 문제 발생.
* **해결:** JavaScript `navigator.mediaDevices.enumerateDevices()` API를 사용해 장치 목록을 스캔한 뒤, 장치 이름(label)에 **'IR'** 또는 **'Virtual'**이 포함된 장치를 **명시적으로 필터링 및 제외**하고 'Webcam' 또는 'Internal' 장치를 우선 연결하는 로직을 구현하여 호환성을 확보했습니다.

---

## 6. 향후 개선 방향

### 6.1 모델 및 서비스 안정화 개선 (Critical Stability)

1.  **신뢰도 임계값 기반 예외 처리 시스템 도입:**
    * **목표:** 신뢰도가 낮은 오분류 가능성을 원천 차단하여 정보의 정확성 확보.
    * **구현:** 모델 예측 결과의 **신뢰도가 특정 기준(예: 60%) 미만**일 경우, 가장 높은 확률의 클래스를 강제 분류하는 대신 `'판독 불가(Unknown)'` 또는 `'재촬영 요청'` 메시지를 반환하는 **Fail-safe 로직**을 프론트엔드 및 백엔드에 통합합니다.

2.  **능동 학습 (Active Learning) 파이프라인 구축:**
    * **목표:** 모델의 성능을 지속적으로 향상시키고, 새로운 데이터 경향에 능동적으로 대응.
    * **구현:** 신뢰도가 낮게 예측된 이미지나 사용자 피드백(오분류 신고) 이미지를 서버에 자동 수집하고, 이를 주기적인 **재학습(Retrain)**에 활용하는 **자동화된 MLOps 파이프라인**을 설계하여 모델의 정적(Static) 한계를 극복합니다. 

### 6.2 기능 확장 및 분류 고도화

3.  **실시간 객체 탐지 (Object Detection) 도입:**
    * **목표:** 이미지 분류를 넘어 여러 쓰레기를 동시에 탐지하고 위치를 파악하는 시스템으로 확장. (예: YOLO, Faster R-CNN 기반)
    * **효과:** 한 장의 사진에 여러 폐기물이 있을 때 모든 폐기물에 대한 분류 결과를 제공할 수 있습니다.

4.  **분류 카테고리 확장:**
    * **목표:** 서비스의 적용 범위를 확대하여 실제 생활의 다양한 폐기물을 커버.
    * **구현:** 의류, 음식물 폐기물, 복합 재질 등 분류 카테고리를 다변화하고 이에 따른 데이터셋을 보강합니다.

---

## 7. 부록 (최종 동작 화면)

### 7.1 서버 실행 성공 화면

<img width="1203" height="1021" alt="8000" src="https://github.com/user-attachments/assets/b4c9892e-562b-4e8a-b0e6-c0adc8ed5a03" />



### 7.2 AI 분류 시스템 웹 UI (메인 화면)

<table>
  <tr>
    <td style="text-align: center; width: 50%; padding: 0 10px;">
      <img width="979" alt="메인 화면 (실시간 카메라 모드)" src="https://github.com/user-attachments/assets/9474f575-482d-4937-9d66-849c089a1838" style="max-width: 100%; height: auto;">
      <p style="font-size: 0.9em; margin-top: 10px; color: #555;">실시간 카메라 </p>
    </td>
    <td style="text-align: center; width: 50%; padding: 0 10px;">
      <img width="780" height="1034" alt="7" src="https://github.com/user-attachments/assets/a4848a47-2074-4da5-b90d-e11644ddb077" style="max-width: 100%; height: auto;">
      <p style="font-size: 0.9em; margin-top: 10px; color: #555;">앨범 업로드 </p>
    </td>
  </tr>
</table>

### 7.3 최종 분류 결과 화면 (문제 해결 증명) 

<img width="868" height="1002" alt="3" src="https://github.com/user-attachments/assets/966eeb20-3549-442a-9b38-19e6be0312b1" />

| 항목 | 내용 |
| :--- | :--- |
| **업로드된 이미지** | 일반쓰레기 봉투 사진 |
| **AI 분석 결과** | **일반쓰레기 (General)** |
| **신뢰도** | **[100]%** |

---

<img width="560" height="659" alt="image (1)" src="https://github.com/user-attachments/assets/221171f5-6f8e-4b5b-a61b-97e1a08bf0fd" />

| 항목 | 내용 |
| :--- | :--- |
| **업로드된 이미지** |  패트병 사진 |
| **AI 분석 결과** | **플라스틱 (Plastic)** |
| **신뢰도** | **[98.1]%** |

---

## 8. 참고 문헌

* Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
* PyTorch Documentation and Tutorials.
* **[주요 데이터 출처]** [(Waste Classification Dataset, Kaggle Repository)](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
* **[ 보강 데이터 출처]** (Google Image Search and Manual Collection for specific General Waste types)



