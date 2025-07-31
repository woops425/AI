# Transformer 기반 손 제스처 분류 전체 코드

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from google.colab import files
import copy

from google.colab import files
uploaded = files.upload()  # 실행 후 .npz 파일 선택해서 업로드

with np.load('processed_typing_data.npz') as data:
    X = data['X_sequences']
    y = data['y_sequences']

# --- 데이터셋 클래스 (기존과 동일) ---
class GestureDataset(Dataset):
    def __init__(self, X_data, y_data, augment=False):
        self.X = torch.from_numpy(X_data).float()
        self.y = torch.from_numpy(y_data).long()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Focal Loss (기존과 동일) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()
    
    # --- Transformer 기반 모델 정의 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, num_classes, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])

# --- 데이터 로딩 및 분할 ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

train_dataset = GestureDataset(X_train, y_train, augment=False)
val_dataset = GestureDataset(X_val, y_val, augment=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- 학습 준비 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = X_train.shape[2]
NUM_CLASSES = len(np.unique(y_train))
model = TransformerClassifier(INPUT_SIZE, NUM_CLASSES).to(device)
criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

uploaded = files.upload()

# --- 학습 루프 ---
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(30):
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

    print(f"Epoch {epoch+1} | Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f} | ROC-AUC: {roc_auc:.4f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        print("  -> Best model updated")

model.load_state_dict(best_model_wts)
print("\n--- 학습 완료 및 최고 성능 모델 로드 ---")

# --- 모델 저장 및 다운로드 ---
MODEL_SAVE_PATH = "transformer_model.pt"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n모델 저장 완료: {MODEL_SAVE_PATH}")

from google.colab import files
files.download(MODEL_SAVE_PATH)