# TCN 기반 손 제스처 분류 전체 코드

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
import copy
from google.colab import drive

from google.colab import files
uploaded = files.upload()  # 실행 후 .npz 파일 선택해서 업로드

with np.load('processed_typing_data.npz') as data:
    X = data['X_sequences']
    y = data['y_sequences']

# 설정값
BATCH_SIZE = 32
AUGMENTATION_PROBABILITY = 0.5
EPOCHS = 50
LEARNING_RATE = 0.001

# --- 증강 함수들 ---
def rotate_landmarks_torch(sequence):
    device = sequence.device
    dtype = sequence.dtype
    angle_x = torch.empty(1, device=device).uniform_(-0.1, 0.1)
    angle_y = torch.empty(1, device=device).uniform_(-0.1, 0.1)
    angle_z = torch.empty(1, device=device).uniform_(-0.1, 0.1)
    cos_x, sin_x = torch.cos(angle_x), torch.sin(angle_x)
    cos_y, sin_y = torch.cos(angle_y), torch.sin(angle_y)
    cos_z, sin_z = torch.cos(angle_z), torch.sin(angle_z)
    Rx = torch.tensor([[1,0,0],[0,cos_x,-sin_x],[0,sin_x,cos_x]], dtype=dtype, device=device).squeeze()
    Ry = torch.tensor([[cos_y,0,sin_y],[0,1,0],[-sin_y,0,cos_y]], dtype=dtype, device=device).squeeze()
    Rz = torch.tensor([[cos_z,-sin_z,0],[sin_z,cos_z,0],[0,0,1]], dtype=dtype, device=device).squeeze()
    rotation_matrix = torch.matmul(Rz, torch.matmul(Ry, Rx))
    seq_reshaped = sequence.view(-1, 21, 3)
    seq_rotated = torch.matmul(seq_reshaped, rotation_matrix)
    return seq_rotated.view(-1, 63)

def scale_landmarks_torch(sequence):
    scale_factor = torch.empty(1, dtype=sequence.dtype, device=sequence.device).uniform_(0.9, 1.1)
    return sequence * scale_factor

def jitter_landmarks_torch(sequence):
    noise = torch.randn_like(sequence) * 0.005
    return sequence + noise

def augment_sequence_torch(sequence):
    if torch.rand(1) < AUGMENTATION_PROBABILITY:
        sequence = rotate_landmarks_torch(sequence)
    if torch.rand(1) < AUGMENTATION_PROBABILITY:
        sequence = scale_landmarks_torch(sequence)
    if torch.rand(1) < AUGMENTATION_PROBABILITY:
        sequence = jitter_landmarks_torch(sequence)
    return sequence

# --- Dataset 클래스 ---
class GestureDataset(Dataset):
    def __init__(self, X_data, y_data, augment=False):
        self.X = torch.from_numpy(X_data).float()
        self.y = torch.from_numpy(y_data).long()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sequence = self.X[idx]
        label = self.y[idx]
        if self.augment:
            sequence = augment_sequence_torch(sequence)
        return sequence, label

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
        # --- TCN 모델 ---
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation  

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        # 여기서 out과 res의 길이가 정확히 같아야 함!
        return F.relu(out[:, :, -res.size(2):] + res)  # output 길이 잘라서 맞추기


class TCNClassifier(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2, num_classes=2):
        super(TCNClassifier, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
          dilation_size = 2 ** i
          in_channels = input_size if i == 0 else num_channels[i - 1]
          out_channels = num_channels[i]
          layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                             dilation=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.network(x)
        return self.fc(y[:, :, -1])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_dataset = GestureDataset(X_train, y_train, augment=True)
val_dataset = GestureDataset(X_val, y_val, augment=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 모델 초기화 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = X_train.shape[2]
NUM_CLASSES = len(np.unique(y_train))
model = TCNClassifier(
    input_size=INPUT_SIZE,
    num_channels=[64, 64, 64],
    kernel_size=3,
    dropout=0.2,
    num_classes=NUM_CLASSES
).to(device)
criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 누적 학습할 경우, 이전에 저장한 모델 가중치 불러오기
model.load_state_dict(torch.load("TCN_model_v2.pt"))
print("이전 모델 가중치 로드 완료")
uploaded = files.upload()

# --- 학습 루프 ---
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
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

    model.eval()
    val_loss, val_corrects = 0.0, 0
    all_labels, all_probs = [], []
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)
            val_loss += loss.item() * sequences.size(0)
            val_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = val_corrects.double() / len(val_dataset)
    all_probs = np.array(all_probs)
    try:
        if NUM_CLASSES == 2:
            roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError:
        roc_auc = 0.0

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f} | ROC-AUC: {roc_auc:.4f}")
    # ROC-AUC 수치 : 모델이 클래스 간 구분을 얼마나 잘 하는지를 0~1 사이 값으로 나타낸 것. 1에 가까울수록 클래스 간 구분이 잘 되는 것임.

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f"  -> Best model updated (val_loss: {best_val_loss:.4f})")

model.load_state_dict(best_model_wts)
print("\n--- 학습 완료 및 최고 성능 모델 로드 ---")

MODEL_SAVE_PATH = "TCN_model_v3.pt"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n모델 저장 완료: {MODEL_SAVE_PATH}")

from google.colab import files
files.download(MODEL_SAVE_PATH)