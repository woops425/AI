from google.colab import files

uploaded = files.upload()

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 데이터 불러오기
X = np.load("X_zonly_sequences.npy")
y = np.load("y_zonly_labels.npy")

# 하이퍼파라미터 설정
INPUT_SIZE = 1
SEQ_LEN = 30
HIDDEN_SIZE = 32
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.003

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 정의
class ZSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_dataset = ZSequenceDataset(X_train, y_train)
val_dataset = ZSequenceDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 모델 정의
class ZLSTMClassifier(nn.Module):
    def __init__(self):
        super(ZLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        out = self.fc(out)
        return out

model = ZLSTMClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 저장된 모델 가중치 업로드
uploaded = files.upload()
model.load_state_dict(torch.load('lstm_of_zOnly_가중치x_v0.pt'))
print("저장된 모델 가중치 불러오기 완료")

# 학습 및 검증 루프
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)

    # 검증
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    val_acc = correct / total * 100
    avg_val_loss = val_loss / len(val_loader)

    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# 모델 저장
torch.save(model.state_dict(), 'lstm_of_zOnly_가중치x_v1.pt')
print("모델 저장 완료")

# Colab에서 다시 다운로드
files.download('lstm_of_zOnly_가중치x_v1.pt')
