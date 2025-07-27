# 1. 기본 설정 및 라이브러리 임포트
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import copy # 모델 가중치 복사를 위해 추가

# Google Drive 마운트 (Colab에서 실행 시)
from google.colab import drive
drive.mount('/content/drive')

# 설정값
NPZ_FILE_PATH = '/content/drive/MyDrive/gesture_data_v2/processed_typing_data.npz' # Google Drive 경로 예시
# NPZ_FILE_PATH = 'processed_typing_data.npz' # 로컬 또는 Colab 세션 저장소 경로
BATCH_SIZE = 32
AUGMENTATION_PROBABILITY = 0.5 # 각 증강을 적용할 확률
EPOCHS = 50 # 학습 에포크 수
LEARNING_RATE = 0.001 # 학습률

# --- PyTorch 버전 증강 함수들 ---

def rotate_landmarks_torch(sequence):
    """(PyTorch 버전) 시퀀스 전체를 랜덤한 각도로 회전"""
    device = sequence.device
    dtype = sequence.dtype

    angle_x = torch.empty(1, device=device).uniform_(-0.1, 0.1)
    angle_y = torch.empty(1, device=device).uniform_(-0.1, 0.1)
    angle_z = torch.empty(1, device=device).uniform_(-0.1, 0.1)

    cos_x, sin_x = torch.cos(angle_x), torch.sin(angle_x)
    cos_y, sin_y = torch.cos(angle_y), torch.sin(angle_y)
    cos_z, sin_z = torch.cos(angle_z), torch.sin(angle_z)

    Rx = torch.tensor([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]], dtype=dtype, device=device).squeeze()
    Ry = torch.tensor([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]], dtype=dtype, device=device).squeeze()
    Rz = torch.tensor([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]], dtype=dtype, device=device).squeeze()

    rotation_matrix = torch.matmul(Rz, torch.matmul(Ry, Rx))

    seq_reshaped = sequence.view(-1, 21, 3)
    seq_rotated = torch.matmul(seq_reshaped, rotation_matrix)
    return seq_rotated.view(-1, 63)

def scale_landmarks_torch(sequence):
    """(PyTorch 버전) 시퀀스 전체의 크기를 랜덤하게 조절"""
    scale_factor = torch.empty(1, dtype=sequence.dtype, device=sequence.device).uniform_(0.9, 1.1)
    return sequence * scale_factor

def jitter_landmarks_torch(sequence):
    """(PyTorch 버전) 시퀀스 전체에 미세한 노이즈 추가"""
    noise = torch.randn_like(sequence) * 0.005
    return sequence + noise

def augment_sequence_torch(sequence):
    """(PyTorch 버전) 증강 기법들을 확률적으로 적용하는 메인 함수"""
    if torch.rand(1) < AUGMENTATION_PROBABILITY:
        sequence = rotate_landmarks_torch(sequence)
    if torch.rand(1) < AUGMENTATION_PROBABILITY:
        sequence = scale_landmarks_torch(sequence)
    if torch.rand(1) < AUGMENTATION_PROBABILITY:
        sequence = jitter_landmarks_torch(sequence)
    return sequence

class GestureDataset(Dataset):
    """
    손 제스처 .npz 파일을 위한 사용자 정의 PyTorch Dataset.
    실시간 데이터 증강을 지원합니다.
    """
    def __init__(self, X_data, y_data, augment=False):
        """
        Args:
            X_data (np.array): 특징 데이터 (시퀀스).
            y_data (np.array): 레이블 데이터.
            augment (bool): True이면 훈련 시점에 데이터 증강을 적용.
        """
        self.X = torch.from_numpy(X_data).float()
        self.y = torch.from_numpy(y_data).long()
        self.augment = augment
        print(f"Dataset 생성 완료. 총 샘플 수: {len(self.X)}, 증강: {self.augment}")

    def __len__(self):
        """데이터셋의 전체 샘플 수를 반환"""
        return len(self.X)

    def __getitem__(self, idx):
        """인덱스(idx)에 해당하는 샘플을 반환"""
        sequence = self.X[idx]
        label = self.y[idx]

        if self.augment:
            sequence = augment_sequence_torch(sequence)

        return sequence, label
    
    # .npz 파일 로드
with np.load(NPZ_FILE_PATH) as data:
    X = data['X_sequences']
    y = data['y_sequences']

# 훈련 데이터와 검증 데이터 분리 (8:2 비율)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"훈련 데이터 형태: {X_train.shape}, {y_train.shape}")
print(f"검증 데이터 형태: {X_val.shape}, {y_val.shape}")

# 훈련용 데이터셋 (증강 적용)
train_dataset = GestureDataset(X_train, y_train, augment=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 검증용 데이터셋 (증강 미적용)
val_dataset = GestureDataset(X_val, y_val, augment=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("\nDataLoader 생성 완료.")

class FocalLoss(nn.Module):
    """
    Focal Loss 함수 구현.
    클래스 불균형이 심한 데이터셋에서 사용하면 효과적입니다.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 모델의 출력 (logits) (batch_size, num_classes)
            targets: 정답 레이블 (batch_size)
        """
        # CrossEntropyLoss 계산 (log_softmax + nll_loss)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 예측 확률(pt) 계산
        pt = torch.exp(-ce_loss)

        # Focal Loss 계산
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
        class LSTMClassifier(nn.Module):
    """
    손 제스처 시퀀스 분류를 위한 LSTM 모델.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.5):
        """
        Args:
            input_size (int): 입력 특징의 수 (e.g., 63 for 21 landmarks * 3 coords).
            hidden_size (int): LSTM hidden state의 차원.
            num_layers (int): LSTM 레이어의 수.
            num_classes (int): 분류할 클래스의 수.
            dropout_prob (float): 드롭아웃 확률.
        """
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # 입력 텐서 형식을 (batch, seq, feature)로 설정
            dropout=dropout_prob if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 시퀀스 텐서 (batch_size, seq_length, input_size)
        """
        # LSTM의 마지막 타임스텝의 출력을 사용
        lstm_out, _ = self.lstm(x) # lstm_out: (batch, seq_len, hidden_size)

        # 마지막 시퀀스의 출력만 선택
        last_output = lstm_out[:, -1, :]

        out = self.dropout(last_output)
        out = self.fc(out)
        return out
    
    # --- 학습 설정 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델 파라미터
INPUT_SIZE = X_train.shape[2]
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = len(np.unique(y_train))
DROPOUT = 0.5

# 모델, 손실 함수, 옵티마이저 초기화
model = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, DROPOUT).to(device)
criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())

print("\n--- 학습 시작 ---")

# --- 학습 및 검증 루프 ---
for epoch in range(EPOCHS):
    # --- 훈련 단계 ---
    model.train()
    train_loss, train_corrects = 0.0, 0

    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * sequences.size(0)
        train_corrects += torch.sum(preds == labels.data)

    epoch_train_loss = train_loss / len(train_dataset)
    epoch_train_acc = train_corrects.double() / len(train_dataset)

    # --- 검증 단계 ---
    model.eval()
    val_loss, val_corrects = 0.0, 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1) # ROC-AUC 계산을 위한 확률값

            val_loss += loss.item() * sequences.size(0)
            val_corrects += torch.sum(preds == labels.data)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = val_corrects.double() / len(val_dataset)

    # ROC-AUC 계산 (Multi-class)
    all_probs = np.array(all_probs)
    try:
      # 클래스가 2개일 경우와 3개 이상일 경우를 다르게 처리
      if NUM_CLASSES == 2:
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
      else:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError:
        roc_auc = 0.0 # 레이블이 하나만 있을 경우 에러 방지

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f} | "
          f"Val ROC-AUC: {roc_auc:.4f}")

    # 최고 성능 모델 저장
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f"  -> Best model updated (val_loss: {best_val_loss:.4f})")

# 최고 성능 모델 가중치 로드
model.load_state_dict(best_model_wts)
print("\n--- 학습 종료 ---")
