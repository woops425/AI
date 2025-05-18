from google.colab import files

uploaded = files.upload()

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

# ✅ 데이터 불러오기 및 Z 좌표 가중치 부여
X = np.load("X_sequences.npy")
y = np.load("y_labels.npy")
z_weight = 1.5
X[:, :, 2] *= z_weight

# ✅ Dataset 정의
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ✅ 모델 정의 (LSTM with hidden_size=256)
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=3, hidden_size=256, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# ✅ Train/Validation 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
train_dataset = SequenceDataset(X_train, y_train)
val_dataset = SequenceDataset(X_val, y_val)

# ✅ 클래스 불균형 대응 (Weighted Sampler)
class_weights = 1. / np.bincount(y_train)
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ✅ 모델 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier().to(device)

# 클래스 1 (입력)에 가중치 부여
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5], dtype=torch.float32).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 저장된 모델 가중치 업로드
uploaded = files.upload()
model.load_state_dict(torch.load('lstm_of_class1_Z_weight_v0.pt'))
print("저장된 모델 가중치 불러오기 완료")

# ✅ 학습 루프
EPOCHS = 30
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # 검증
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output = model(batch_X)
            preds = torch.argmax(output, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {total_loss / len(train_loader):.4f} | Val Acc: {correct / total:.4f}")

# 모델 저장
torch.save(model.state_dict(), 'lstm_of_class1_Z_weight_v1.pt')
print("모델 저장 완료")

# Colab에서 다시 다운로드
files.download('lstm_of_class1_Z_weight_v1.pt')