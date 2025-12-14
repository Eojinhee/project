import os

# [중요] 그래픽 카드 라이브러리 충돌 방지 (WinError 1114 해결용)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 학습률 조절기 추가
import time

# ==========================================
# [설정 영역]
# ==========================================
DATA_DIR = './dataset'
SAVE_PATH = 'waste_model.pth'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20  # 조기 종료(Early Stopping)를 위해 넉넉하게 20으로 설정
NUM_CLASSES = 5


# ==========================================


def train_model():
    print(f" AI 학습을 시작합니다! (설정: {EPOCHS} 에폭, Early Stopping 활성화)")
    print(" 이번 학습은 조명/배경 변화에 덜 민감하도록 데이터 증강이 강화됩니다.")

    # 1. 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 사용할 장치: {device}")

    # 2. 이미지 전처리 (데이터 증강 강화)
    # ------------------------------------------------------------------
    # ⭐ [핵심 수정] train_transform: ColorJitter 강도를 높여 배경 의존성을 줄임
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # 밝기, 대비, 채도를 무작위로 30%까지 변화시켜 조명 환경 다양화
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # val_transform: 검증 시에는 증강 없이 기본 전처리만 수행
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # ------------------------------------------------------------------

    # 3. 데이터셋 및 데이터 로더 준비
    if not os.path.exists(DATA_DIR):
        print(f" 오류: 데이터 폴더 '{DATA_DIR}'를 찾을 수 없습니다. 폴더 경로를 확인해주세요.")
        return

    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)

    # 데이터 분할 (훈련 80%, 검증 20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 검증 데이터셋에만 검증용 전처리 적용 ( train_transform을 val_transform으로 교체)
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f" 데이터셋 준비 완료: 훈련 {len(train_dataset)}개, 검증 {len(val_dataset)}개")
    print(f"클래스 목록: {full_dataset.classes}")

    # 4. 모델 설정 (EfficientNet B0 전이 학습)
    # 최신 가중치 사용
    try:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except:
        # 최신 가중치 로드가 실패하면 이전 방식 사용 (이전 코드를 기반으로 호환성 유지)
        model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')

    # 분류기 레이어 교체
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    model.to(device)

    # 5. 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 학습률 스케줄러 (검증 정확도가 정체되면 학습률 감소)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    # 6. 모델 학습
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")

        # [훈련 모드]
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f"   [훈련] 오차: {running_loss / len(train_loader):.4f}, 정확도: {train_acc:.2f}%")

        # [검증 모드] (최고 성능 저장 로직)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"   [검증] 정확도: {val_acc:.2f}%")

        # 학습률 스케줄러 업데이트
        scheduler.step(val_acc)

        #  조기 종료 로직: 최고 정확도 갱신 시 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"    최고 정확도 갱신! 모델 저장 ({best_acc:.2f}%) ")
            torch.save(model.state_dict(), SAVE_PATH)

        # 조기 종료 조건 (학습률이 너무 낮아지면 종료)
        if optimizer.param_groups[0]['lr'] < 1e-6:  # 학습률이 1e-6보다 낮아지면
            print(" 학습률이 최저치에 도달하여 학습을 조기 종료합니다.")
            break

    end_time = time.time()
    print(f"\n--- 학습 종료 ---")
    print(f"최고 검증 정확도: {best_acc:.2f}%")
    print(f"총 소요 시간: {end_time - start_time:.2f}초")


if __name__ == '__main__':
    train_model()