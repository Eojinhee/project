from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from pydantic import BaseModel
import time # 시뮬레이션 지연을 위해 필요
from PIL import Image # 이미지 파일 처리를 위해 필요

# --------------------
# 1. FastAPI 애플리케이션 초기화
# --------------------
app = FastAPI(
    title="AI 쓰레기 분류 백엔드",
    description="프론트엔드 (index.html)에 분류 결과를 제공하는 API 서버입니다."
)

# --------------------
# 2. CORS 설정 (필수! - 로컬 오류 해결)
# --------------------
# Failed to fetch 오류를 해결하기 위해 로컬 테스트에서는 모든 Origin을 허용합니다.
# 실제 서비스 환경에서는 보안을 위해 origins 리스트에 특정 URL만 지정해야 합니다.
origins = ["*"] # 모든 Origin 허용 (로컬 테스트용)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# 3. 데이터 모델 정의
# --------------------

class Prediction(BaseModel):
    label: str
    confidence: float

class ClassificationResult(BaseModel):
    status: str
    predictions: List[Prediction]
    model_version: str = "EfficientNet_V2.1"


# --------------------
# 4. 분류 엔드포인트 정의
# --------------------

@app.post("/classify", response_model=ClassificationResult)
async def classify_image(file: UploadFile = File(...)):
    """
    업로드된 이미지 파일을 받아 분류 결과를 반환합니다.
    실제 AI 모델 추론 코드가 들어갈 위치입니다.
    """
    
    # [1] 파일 확인 및 처리
    try:
        # 파일명을 기반으로 시뮬레이션 결과를 조정하기 위해 파일명을 사용합니다.
        file_name = file.filename if file.filename else "unknown_file"
        
        # 실제 환경이라면 여기서 PIL로 이미지를 열고 모델에 전달합니다.
        # contents = await file.read()
        # image = Image.open(io.BytesIO(contents))
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid file upload or file processing error: {e}")

    # [2] 실제 모델 추론 대신 시뮬레이션 데이터 반환
    # index.html의 simulateClassification 함수와 동일한 로직을 재현하여 
    # 백엔드와 프론트엔드의 데이터 구조가 일치함을 보장합니다.
    
    # 1초 지연을 추가하여 AI가 실제로 분석하는 듯한 느낌을 줍니다.
    time.sleep(1.0) 
    
    # 파일명을 기반으로 분류 결과 시뮬레이션 (index.html의 로직과 동일)
    lower_case_name = file_name.lower()
    labels = ['플라스틱', '종이', '캔', '유리', '일반쓰레기'] 
    
    if '종이' in lower_case_name or 'paper' in lower_case_name or '박스' in lower_case_name:
        primary_label = '종이'
    elif '캔' in lower_case_name or 'can' in lower_case_name or '콜라' in lower_case_name:
        primary_label = '캔'
    elif '유리' in lower_case_name or ('병' in lower_case_name and 'pet' not in lower_case_name and '플라스틱' not in lower_case_name):
        primary_label = '유리'
    elif '플라스틱' in lower_case_name or 'pet' in lower_case_name or '페트' in lower_case_name or 'bottle' in lower_case_name:
        primary_label = '플라스틱' 
    elif '쓰레기' in lower_case_name or 'general' in lower_case_name or '오염' in lower_case_name or '비닐' in lower_case_name:
        primary_label = '일반쓰레기'
    else:
        ambiguous_labels = ['플라스틱', '유리', '일반쓰레기']
        # 백엔드에서는 random 모듈이 없으므로, 단순 임시 값 사용
        primary_label = ambiguous_labels[int(time.time()) % len(ambiguous_labels)]


    predictions_list = []
    # Top 1 신뢰도: 90% ~ 99% 사이의 값으로 설정
    primary_confidence = 0.90 + (int(time.time() * 1000) % 100) / 1000.0 
    predictions_list.append({"label": primary_label, "confidence": primary_confidence})

    # 나머지 신뢰도 할당 (정렬을 위해 임시로 낮은 값 추가)
    for label in labels:
        if label != primary_label:
            confidence = (int(time.time() * 100) % 10) / 100.0 # 0% ~ 10%
            predictions_list.append({"label": label, "confidence": confidence})

    # 신뢰도 순으로 정렬하여 Top 3가 되도록 준비
    predictions_list.sort(key=lambda x: x['confidence'], reverse=True)

    # [3] 결과 반환
    return ClassificationResult(
        status="success",
        predictions=predictions_list,
        model_version="EfficientNet_V2.1"
    )


