import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from PIL import Image
from io import BytesIO

# [중요] 그래픽 카드 라이브러리 충돌 방지 (WinError 1114 해결용)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ==========================================
# [설정 영역]
# ==========================================
MODEL_PATH = 'waste_model.pth'  # 저장된 모델 파일 이름
NUM_CLASSES = 5
# 클래스 이름 순서를 학습 폴더 순서와 명확히 일치시킵니다.
CLASS_NAMES = ['can', 'general', 'glass', 'paper', 'plastic']
DEFAULT_PORT = 8000  # 포트를 8000번으로 재설정
# ==========================================

# 1. 장치 설정 및 FastAPI 앱 초기화
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI()
model = None  # 모델을 전역 변수로 선언

# 2. 이미지 전처리 (trainer.py의 val_transform과 동일해야 함)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 3. 모델 로드 함수 (서버 시작 시 한 번만 실행)
@app.on_event("startup")
async def load_model():
    global model
    try:
        # 모델 구조 정의 (EfficientNet B0)
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)

        # 학습된 weight 로드
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint)

        model.to(DEVICE)
        model.eval()  # 평가 모드로 설정
        print(f" AI 모델 로드 성공! ({DEVICE} 모드)")

    except FileNotFoundError:
        print(f" 오류: 모델 파일 '{MODEL_PATH}'를 찾을 수 없습니다. trainer.py를 먼저 실행하세요.")
        model = None
    except Exception as e:
        print(f" 모델 로드 중 심각한 오류 발생: {e}")
        model = None


# 4. 이미지 예측 함수
def predict_image(image: Image.Image):
    if model is None:
        return "error", 0.0, "모델 로드 실패"

    # 이미지 전처리 및 배치 차원 추가
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)

        # 가장 높은 확률과 해당 인덱스 찾기
        conf, predicted_index = torch.max(probabilities, 1)

        # 예측된 클래스 이름과 신뢰도: CLASS_NAMES 배열의 인덱스를 사용
        predicted_class = CLASS_NAMES[predicted_index.item()]
        confidence = conf.item() * 100

    return predicted_class, confidence, "분석 완료"


# ==========================================
# 5. API 엔드포인트
# ==========================================

# 루트 엔드포인트: index.html 파일 제공
@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html 파일을 찾을 수 없습니다.</h1>", status_code=404)


# 예측 API 엔드포인트
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse({"predicted_class": "error", "confidence": 0.0, "message": "AI 모델이 로드되지 않았습니다."},
                            status_code=500)

    # PIL Image 객체로 이미지 로드 (RGB 3채널 보장)
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    # 예측 실행
    predicted_class, confidence, message = predict_image(image)

    return JSONResponse({
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "message": message
    })


# 6. Uvicorn 실행
if __name__ == "__main__":
    # 호스트 주소를 '127.0.0.1'로 명시하고 포트 8000 사용
    uvicorn.run(app, host="127.0.0.1", port=DEFAULT_PORT)