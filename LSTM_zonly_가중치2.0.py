# 2025.05.25
# Z값만, 클래스 1에 가중치 2.0배

from google.colab import files

uploaded = files.upload()

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# 데이터 로딩
X = np.load("X_zonly_sequences.npy")
y = np.load("y_zonly_labels.npy")

# 커스텀 데이터셋 정의
class ZOnlySequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ✅ LSTM 모델 정의
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out.squeeze(1)

# 데이터 분할 및 로더 설정
dataset = ZOnlySequenceDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

# 클래스 가중치 수동 설정 (label 0, label 1)
class_weights = torch.tensor([1.0, 2.0], dtype=torch.float32)

# 모델, 손실함수, 옵티마이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier().to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

# 기존 가중치 불러오기
uploaded = files.upload()
model.load_state_dict(torch.load('lstm_zOnly_class1_weight_v24.pt'))
print("저장된 모델 가중치 불러오기 완료")

# 학습 함수 정의 
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=30):
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.sigmoid(outputs) >= 0.5
                correct += (preds == labels.bool()).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss = {total_loss/len(train_loader):.4f}, "
              f"Val Loss = {val_loss/len(val_loader):.4f}, Val Acc = {val_acc:.4f}")
        
# 학습 실행
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=30)

# 모델 저장
torch.save(model.state_dict(),
           'lstm_zOnly_class1_weight_v25.pt')
print("모델 저장 완료")

# Colab 다운로드
files.download('lstm_zOnly_class1_weight_v25.pt')