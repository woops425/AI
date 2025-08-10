# ##############################################################################
# @title 셀 1 (수정됨): 기본 설정 및 라이브러리 임포트
# ##############################################################################
import os
import glob
import pandas as pd
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 데이터 불균형 처리를 위한 라이브러리
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# 재현성을 위한 시드 고정
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {device}")
print("-" * 50)

# 보고서에서 권장하는 하이퍼파라미터들을 설정합니다.
class Config:
    # --- 데이터 관련 ---
    SEQUENCE_LENGTH = 20

    # --- 훈련 관련 (보고서 표 4 참조) ---
    OPTIMIZER = 'AdamW'
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64

    # [수정 사항] 기준 모델과 최종 모델의 에포크 수를 분리하여 정의
    # 기준 모델은 가이드 역할만 하므로 적당히 훈련
    BASELINE_EPOCHS = 20
    # 최종 모델은 최고 성능을 찾기 위해 더 많이 훈련
    FINAL_EPOCHS = 50

    # --- 모델 아키텍처 관련 (TCN, 보고서 표 4 참조) ---
    TCN_INPUT_CHANNELS = -1
    TCN_NUM_CHANNELS = [128] * 2
    TCN_KERNEL_SIZE = 3
    DROPOUT = 0.2

# 설정값 인스턴스 생성
config = Config()

# ##############################################################################
# @title 셀 2 (수정됨): 데이터 로드 및 시퀀스 데이터 생성
# ##############################################################################

# 데이터가 저장된 폴더 경로
# Colab에 'gesture_data_v3' 폴더를 만들고 그 안에 CSV를 업로드했다고 가정합니다.

# Google Drive 마운트 (Colab에서 실행 시)
from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = '/content/drive/MyDrive/gesture_data_v3/'
file_pattern = "hybrid_typing_data_v3_*.csv"
all_files = glob.glob(os.path.join(DATA_DIR, file_pattern))

if not all_files:
    print(f"'{DATA_DIR}' 폴더에 '{file_pattern}' 패턴과 일치하는 데이터 파일이 없습니다.")
    print("Colab 왼편의 파일 탐색기를 통해 데이터를 업로드했는지 확인해주세요.")
else:
    # 1. 모든 CSV 파일을 하나로 합치기
    df_list = [pd.read_csv(f) for f in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"총 {len(all_files)}개의 파일에서 {len(combined_df)}개의 프레임 데이터를 불러왔습니다.")
    print(f"초기 레이블 분포:\n{combined_df['label'].value_counts()}\n")

    # 2. 피처(X)와 레이블(y) 분리
    X_raw = combined_df.drop('label', axis=1).values

    # [수정 사항] 레이블 데이터를 정수(int) 타입으로 명시적으로 변환합니다.
    y_raw = combined_df['label'].values.astype(int)

    # 3. 시퀀스 데이터 생성 함수
    def create_sequences(X, y, sequence_length):
        """연속된 프레임 데이터를 시퀀스 데이터로 변환합니다."""
        X_seq, y_seq = [], []
        # 전체 데이터를 순회하며 시퀀스 길이만큼 데이터를 묶습니다.
        for i in range(len(X) - sequence_length + 1):
            X_seq.append(X[i:i + sequence_length])
            # 레이블은 시퀀스의 마지막 프레임의 레이블을 사용합니다.
            y_seq.append(y[i + sequence_length - 1])
        return np.array(X_seq), np.array(y_seq)

    # 4. 시퀀스 데이터 생성 실행
    X_sequences, y_sequences = create_sequences(X_raw, y_raw, config.SEQUENCE_LENGTH)

    # 5. TCN 모델의 입력 채널 수를 데이터에 맞게 동적으로 설정
    config.TCN_INPUT_CHANNELS = X_sequences.shape[2]

    print("시퀀스 데이터 생성 완료!")
    print(f"X 형태: {X_sequences.shape}")
    print(f"y 형태: {y_sequences.shape}")
    print(f"변환 후 레이블 분포: {Counter(y_sequences)}")
    print(f"TCN 입력 채널 수 설정: {config.TCN_INPUT_CHANNELS}")

# ##############################################################################
# @title 셀 3 (수정됨): 공간적 데이터 증강 및 Custom Dataset 정의
# ##############################################################################

# --- 공간적 데이터 증강 함수들 ---
# 보고서 표 3의 가이드를 기반으로 함수를 정의합니다.
# 실제 3D 랜드마크의 특성에 맞게 세부 로직을 수정하여 사용할 수 있습니다.

def augment_rotate(sequence: np.ndarray, max_angle_deg: float = 10.0) -> np.ndarray:
    """
    3D 공간에서 시퀀스를 무작위로 회전시킵니다.
    [수정 사항] np.einsum 대신 np.dot을 사용하고, 데이터 차원을 명시적으로 처리합니다.
    """
    # (Seq_Len, 63) 형태의 원본 시퀀스
    original_shape = sequence.shape
    if original_shape[1] % 3 != 0:
        # 피처 수가 3으로 나누어 떨어지지 않으면 3D 좌표가 아니므로 회전하지 않음
        return sequence

    # (Seq_Len, 63) -> (Seq_Len, 21, 3)으로 변환 (3D 좌표 21개라고 가정)
    num_points = original_shape[1] // 3
    sequence_reshaped = sequence.reshape(original_shape[0], num_points, 3)

    angle_rad = np.random.uniform(-max_angle_deg, max_angle_deg) * (np.pi / 180.0)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Z축 회전 행렬 (3, 3)
    # 필요시 X, Y축 회전 행렬을 추가로 곱하여 복합적인 회전을 구현할 수 있습니다.
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])

    # np.dot을 사용하여 회전 적용
    # (Seq_Len, 21, 3)과 (3, 3) 행렬을 곱합니다.
    rotated_sequence_reshaped = np.dot(sequence_reshaped, rotation_matrix)

    # 다시 원래 형태 (Seq_Len, 63)으로 변환
    rotated_sequence = rotated_sequence_reshaped.reshape(original_shape)

    return rotated_sequence

def augment_scale(sequence: np.ndarray, min_scale: float = 0.9, max_scale: float = 1.1) -> np.ndarray:
    """시퀀스의 크기를 무작위로 조절합니다."""
    scale = np.random.uniform(min_scale, max_scale)
    return sequence * scale

def augment_jitter(sequence: np.ndarray, std: float = 0.01) -> np.ndarray:
    """시퀀스의 각 좌표에 가우시안 노이즈를 추가합니다."""
    noise = np.random.normal(loc=0.0, scale=std, size=sequence.shape)
    return sequence + noise

# --- 온라인 증강을 적용하는 Custom Dataset 클래스 ---
class GestureDataset(Dataset):
    def __init__(self, X, y, sequence_length, apply_augmentations=False):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        self.apply_augmentations = apply_augmentations

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 원본 데이터는 numpy 형태이므로, 텐서로 변환
        x_item = self.X[idx].copy() # 원본 수정을 막기 위해 복사
        y_item = self.y[idx]

        # 훈련(train) 모드일 때만 데이터 증강을 적용
        if self.apply_augmentations:
            # 보고서 표 3의 증강 기법들을 순차적으로 적용
            x_item = augment_rotate(x_item)
            x_item = augment_scale(x_item)
            x_item = augment_jitter(x_item)

        # 최종적으로 PyTorch 텐서로 변환
        x_tensor = torch.FloatTensor(x_item)
        y_tensor = torch.LongTensor([y_item]).squeeze() # 스칼라 텐서로 변환

        return x_tensor, y_tensor

print("공간적 데이터 증강 함수 및 Custom Dataset 클래스가 정의되었습니다. (수정 완료)")

# ##############################################################################
# @title 셀 4 (수정됨): TCN 모델 아키텍처 정의
# ##############################################################################

class TemporalBlock(nn.Module):
    """
    TCN의 기본 구성 요소인 Temporal Block입니다.
    [수정 사항] 컨볼루션 이후 시퀀스 길이를 원본과 동일하게 맞추기 위해 슬라이싱 로직을 추가합니다.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # nn.utils.weight_norm을 conv 레이어에 직접 적용합니다.
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.padding1 = padding # 슬라이싱에 사용할 패딩 값을 저장
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.padding2 = padding # 슬라이싱에 사용할 패딩 값을 저장
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 다운샘플링 레이어는 입력과 출력의 채널 수가 다를 때 사용됩니다.
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (Batch, Channels, Seq_Len)

        # 첫 번째 컨볼루션 블록
        out = self.conv1(x)
        # 늘어난 패딩만큼 끝부분을 잘라내어 길이를 맞춤 (Causal Convolution)
        out = out[:, :, :-self.padding1]
        out = self.relu1(out)
        out = self.dropout1(out)

        # 두 번째 컨볼루션 블록
        out = self.conv2(out)
        out = out[:, :, :-self.padding2]
        out = self.relu2(out)
        out = self.dropout2(out)

        # 잔차 연결 (Residual Connection)
        res = x if self.downsample is None else self.downsample(x)

        # 이제 out과 res의 시퀀스 길이가 동일하므로 덧셈이 가능합니다.
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network 모델입니다.
    보고서 표 4의 하이퍼파라미터를 사용하여 구성됩니다.
    """
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # 인과적 컨볼루션을 위한 패딩 계산
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # 입력: (Batch, Seq_Len, Features)
        # TCN은 (Batch, Features, Seq_Len) 형태를 기대하므로 차원 변경
        x = x.permute(0, 2, 1)

        out = self.tcn(x)

        # 마지막 타임스텝의 출력만 사용하여 분류
        out = self.linear(out[:, :, -1])
        return out

print("TCN 모델 아키텍처가 정의되었습니다. (수정 완료)")

# ##############################################################################
# @title 셀 5: 훈련 및 평가 함수 정의
# ##############################################################################
from tqdm.auto import tqdm # 진행 상황을 시각적으로 보여주는 라이브러리

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    """한 에포크 동안 모델을 훈련시키는 함수"""
    model.train()
    total_loss = 0

    for inputs, labels in tqdm(data_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device):
    """모델의 성능을 평가하는 함수"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)

    # 보고서에서 권장하는 F1-Score를 포함한 평가지표 계산 [cite: 91]
    metrics = {
        'loss': avg_loss,
        'f1_score': f1_score(all_labels, all_preds, average='binary'),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='binary'),
        'recall': recall_score(all_labels, all_preds, average='binary')
    }

    return metrics

print("훈련 및 평가 함수가 정의되었습니다.")

# ##############################################################################
# @title 셀 6 (수정됨): 🚀 1단계 - 기준 모델 훈련 실행
# ##############################################################################
print("--- 1단계: 기준 모델(Baseline Model) 훈련 시작 ---")

# 1. 데이터를 훈련셋과 검증셋으로 분리 (80:20)
X_train, X_val, y_train, y_val = train_test_split(
    X_sequences, y_sequences,
    test_size=0.2,
    random_state=SEED,
    stratify=y_sequences
)

# 2. 훈련용/검증용 Dataset 및 DataLoader 생성
train_dataset = GestureDataset(X_train, y_train, config.SEQUENCE_LENGTH, apply_augmentations=True)
val_dataset = GestureDataset(X_val, y_val, config.SEQUENCE_LENGTH, apply_augmentations=False)
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
print(f"훈련 데이터: {len(train_dataset)}개, 검증 데이터: {len(val_dataset)}개")

# 3. 모델, 손실 함수, 옵티마이저 정의
baseline_model = TCN(
    input_size=config.TCN_INPUT_CHANNELS,
    output_size=2,
    num_channels=config.TCN_NUM_CHANNELS,
    kernel_size=config.TCN_KERNEL_SIZE,
    dropout=config.DROPOUT
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=config.LEARNING_RATE)

print("\n모델 아키텍처:")
print(baseline_model)
print(f"\n총 {config.BASELINE_EPOCHS} 에포크 동안 기준 모델 훈련을 시작합니다...")

# 4. 훈련 루프 실행
# [수정 사항] config.BASELINE_EPOCHS 사용
for epoch in range(config.BASELINE_EPOCHS):
    train_loss = train_one_epoch(baseline_model, train_loader, criterion, optimizer, device)
    val_metrics = evaluate(baseline_model, val_loader, criterion, device)
    print(
        f"Epoch {epoch+1}/{config.BASELINE_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val F1: {val_metrics['f1_score']:.4f}"
    )

# 5. 훈련된 기준 모델 저장
BASELINE_MODEL_PATH = 'baseline_model_tcn.pt'
torch.save(baseline_model.state_dict(), BASELINE_MODEL_PATH)

print("-" * 50)
print(f"1단계 완료: 기준 모델이 '{BASELINE_MODEL_PATH}'에 성공적으로 저장되었습니다.")

# ##############################################################################
# @title 셀 7: T-SMOTE 알고리즘 함수 정의
# ##############################################################################

def t_smote(X, y, model, beta_param_a=2.0, beta_param_b=2.0):
    """
    보고서 및 원본 논문에 기반한 T-SMOTE 알고리즘 구현체입니다.
    X는 (샘플 수, 피처 수) 형태의 2D 배열을 가정합니다.
    시퀀스 데이터의 경우, (샘플 수, 시퀀스 길이 * 피처 수)로 변환하여 입력해야 합니다.

    Args:
        X (np.array): 피처 데이터 (샘플 수, 피처 수)
        y (np.array): 레이블 데이터 (샘플 수,)
        model: 사전 훈련된 분류기. `predict_proba` 메서드가 있어야 합니다.
        beta_param_a (float): 베타 분포의 알파 파라미터
        beta_param_b (float): 베타 분포의 베타 파라미터

    Returns:
        X_resampled (np.array): T-SMOTE로 증강된 피처 데이터
        y_resampled (np.array): T-SMOTE로 증강된 레이블 데이터
    """
    stats = Counter(y)
    minority_class = min(stats, key=stats.get)
    majority_class = max(stats, key=stats.get)

    n_samples_to_generate = stats[majority_class] - stats[minority_class]

    if n_samples_to_generate <= 0:
        print("데이터가 이미 균형을 이루고 있습니다.")
        return X, y

    print(f"T-SMOTE 시작: 소수 클래스({minority_class}) 샘플 {n_samples_to_generate}개를 생성합니다.")

    X_minority = X[y == minority_class]

    # 1단계: 경계 근접 후보 생성 (Boundary Proximity Candidate Generation)
    print("1단계: 경계 근접 후보를 생성합니다...")
    # 모델의 예측 확률을 사용하여 경계에 가까운 샘플(후보)을 식별
    pred_scores = model.predict_proba(X_minority)[:, minority_class]
    # 점수가 0.5에 가까울수록 가중치가 높아짐
    candidate_weights = 1 - np.abs(pred_scores - 0.5)

    # 시간적 인접성을 고려하기 위해 각 소수점 샘플의 최근접 이웃을 찾음
    nn = NearestNeighbors(n_neighbors=2, n_jobs=-1)
    nn.fit(X_minority)

    synthetic_samples = []

    # 2단계 & 3단계: 합성 및 가중 샘플링 (Synthesis & Weighted Sampling)
    print("2단계 및 3단계: 샘플 합성 및 가중 샘플링을 시작합니다...")

    # 가중치를 정규화하여 샘플링 확률로 사용
    sampling_probabilities = candidate_weights / np.sum(candidate_weights)

    # 가중치에 따라 n_samples_to_generate개의 원본 샘플을 복원추출
    selected_indices = np.random.choice(
        np.arange(len(X_minority)),
        size=n_samples_to_generate,
        p=sampling_probabilities,
        replace=True
    )

    for i in tqdm(selected_indices, desc="Synthesizing Samples"):
        base_sample = X_minority[i]
        _, neighbor_indices = nn.kneighbors([base_sample])
        neighbor_sample = X_minority[neighbor_indices[0, 1]]

        # 베타 분포에 따라 가중 내삽하여 새로운 샘플 합성
        beta = np.random.beta(beta_param_a, beta_param_b)
        new_sample = base_sample + beta * (neighbor_sample - base_sample)
        synthetic_samples.append(new_sample)

    if not synthetic_samples:
        print("합성된 샘플이 없습니다.")
        return X, y

    X_resampled = np.vstack([X, np.array(synthetic_samples)])
    y_resampled = np.hstack([y, np.full(n_samples_to_generate, minority_class)])

    print("\nT-SMOTE 완료!")
    print(f"이전 데이터 분포: {stats}")
    print(f"이후 데이터 분포: {Counter(y_resampled)}")

    return X_resampled, y_resampled

print("T-SMOTE 알고리즘 함수가 정의되었습니다.")

# ##############################################################################
# @title 셀 8: PyTorch 모델 래퍼(Wrapper) 정의
# ##############################################################################

class PyTorchModelWrapper:
    """
    PyTorch 모델을 t_smote 함수와 호환되도록 감싸는 래퍼 클래스입니다.
    """
    def __init__(self, model_path, model_class, config, device):
        self.config = config
        self.device = device

        # 모델 클래스를 사용하여 구조를 초기화
        self.model = model_class(
            input_size=config.TCN_INPUT_CHANNELS,
            output_size=2,
            num_channels=config.TCN_NUM_CHANNELS,
            kernel_size=config.TCN_KERNEL_SIZE,
            dropout=config.DROPOUT
        ).to(device)

        # 저장된 1단계 모델의 가중치를 불러옴
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        # 반드시 평가 모드로 설정해야 드롭아웃 등이 비활성화됨
        self.model.eval()
        print(f"'{model_path}'에서 모델을 불러와 평가 모드로 설정했습니다.")

    def predict_proba(self, X_numpy):
        """
        Numpy 배열을 입력받아 예측 확률을 Numpy 배열로 반환합니다.
        T-SMOTE는 2D 데이터를 기대하므로, 시퀀스를 2D로 펼쳐서 예측합니다.
        """
        # (Samples, Seq_Len, Features) -> (Samples, Seq_Len * Features)
        if X_numpy.ndim == 3:
            X_numpy = X_numpy.reshape(X_numpy.shape[0], -1)

        # 각 샘플을 다시 시퀀스 형태로 복원
        # (Samples, Seq_Len * Features) -> (Samples, Seq_Len, Features)
        X_reshaped = X_numpy.reshape(-1, self.config.SEQUENCE_LENGTH, self.config.TCN_INPUT_CHANNELS)
        X_tensor = torch.FloatTensor(X_reshaped).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)
            # 로짓(logit)을 소프트맥스를 통해 확률로 변환
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        return probabilities

print("PyTorch 모델 래퍼 클래스가 정의되었습니다.")

# ##############################################################################
# @title 셀 9: 2단계 - T-SMOTE 데이터셋 생성 실행
# ##############################################################################
print("--- 2단계: T-SMOTE 데이터셋 생성 시작 ---")

# 1. 1단계에서 훈련된 기준 모델을 래퍼 클래스로 불러오기
trained_model_wrapper = PyTorchModelWrapper(
    model_path=BASELINE_MODEL_PATH,
    model_class=TCN,
    config=config,
    device=device
)

# 2. T-SMOTE에 입력하기 위해 시퀀스 데이터를 2D로 펼치기
# (Samples, Seq_Len, Features) -> (Samples, Seq_Len * Features)
X_sequences_flat = X_sequences.reshape(X_sequences.shape[0], -1)

# 3. T-SMOTE 실행
# 입력: 펼쳐진 2D 원본 데이터, 원본 레이블, 훈련된 모델 래퍼
X_resampled_flat, y_resampled = t_smote(
    X_sequences_flat,
    y_sequences,
    model=trained_model_wrapper
)

# 4. 생성된 데이터를 다시 시퀀스 형태로 복원
# (Samples, Seq_Len * Features) -> (Samples, Seq_Len, Features)
X_resampled = X_resampled_flat.reshape(
    -1,
    config.SEQUENCE_LENGTH,
    config.TCN_INPUT_CHANNELS
)

# 5. 생성된 균형 데이터셋을 .npz 파일로 저장
BALANCED_DATASET_PATH = 'balanced_dataset_tcn.npz'
np.savez_compressed(
    BALANCED_DATASET_PATH,
    X=X_resampled,
    y=y_resampled
)

print("-" * 50)
print(f"X Resampled Shape: {X_resampled.shape}")
print(f"y Resampled Shape: {y_resampled.shape}")
print(f"2단계 완료: 균형 데이터셋이 '{BALANCED_DATASET_PATH}'에 성공적으로 저장되었습니다.")

# ##############################################################################
# @title 셀 10: 3단계 - 최종 훈련 준비
# ##############################################################################
print("--- 3단계: 최종 모델 훈련 준비 ---")

# 1. 2단계에서 저장한 균형 데이터셋 불러오기
balanced_data = np.load(BALANCED_DATASET_PATH)
X_balanced = balanced_data['X']
y_balanced = balanced_data['y']

print(f"불러온 균형 데이터셋 레이블 분포: {Counter(y_balanced)}")

# 2. 균형 데이터셋을 최종 훈련셋과 검증셋으로 분리 (80:20)
# 데이터가 이미 균형을 이루므로 stratify 옵션은 필수가 아니지만, 명시적으로 사용합니다.
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_balanced, y_balanced,
    test_size=0.2,
    random_state=SEED,
    stratify=y_balanced
)

# 3. 훈련용 Dataset은 증강을 적용(True), 검증용은 미적용(False)
final_train_dataset = GestureDataset(X_train_final, y_train_final, config.SEQUENCE_LENGTH, apply_augmentations=True)
final_val_dataset = GestureDataset(X_val_final, y_val_final, config.SEQUENCE_LENGTH, apply_augmentations=False)

final_train_loader = DataLoader(final_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
final_val_loader = DataLoader(final_val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

print(f"최종 훈련 데이터: {len(final_train_dataset)}개, 최종 검증 데이터: {len(final_val_dataset)}개")

# ##############################################################################
# @title 셀 11 (수정됨): 3단계 - 최종 모델 훈련 실행
# ##############################################################################
print("--- 3단계: 최종 모델(Final Model) 훈련 시작 ---")

# 1. 최종 모델을 새롭게 초기화
final_model = TCN(
    input_size=config.TCN_INPUT_CHANNELS,
    output_size=2,
    num_channels=config.TCN_NUM_CHANNELS,
    kernel_size=config.TCN_KERNEL_SIZE,
    dropout=config.DROPOUT
).to(device)

# 2. 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(final_model.parameters(), lr=config.LEARNING_RATE)

print("\n최종 모델 아키텍처:")
print(final_model)
print(f"\n총 {config.FINAL_EPOCHS} 에포크 동안 최종 모델 훈련을 시작합니다...")

# 3. 훈련 루프 실행
best_f1_score = 0.0
FINAL_MODEL_PATH = 'final_model_tcn.pt'

# [수정 사항] config.FINAL_EPOCHS 사용
for epoch in range(config.FINAL_EPOCHS):
    train_loss = train_one_epoch(final_model, final_train_loader, criterion, optimizer, device)
    val_metrics = evaluate(final_model, final_val_loader, criterion, device)

    print(
        f"Epoch {epoch+1}/{config.FINAL_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val F1: {val_metrics['f1_score']:.4f} | "
        f"Val Acc: {val_metrics['accuracy']:.4f}"
    )

    if val_metrics['f1_score'] > best_f1_score:
        best_f1_score = val_metrics['f1_score']
        torch.save(final_model.state_dict(), FINAL_MODEL_PATH)
        print(f"  -> New best model saved with F1-Score: {best_f1_score:.4f}")

print("-" * 50)
print(f"3단계 완료: 최종 모델이 '{FINAL_MODEL_PATH}'에 성공적으로 저장되었습니다.")
print(f"최고 검증 F1-Score: {best_f1_score:.4f}")

