from google.colab import files

uploaded = files.upload()

# 라이브러리 임포트
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from google.colab import files


# 1. 하이퍼파라미터 설정
SEQ_LEN = 30
INPUT_SIZE = 30
HIDDEN_SIZE = 128
NUM_CLASSES = 2
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.0003

# 2. 데이터 로딩 및 정규화
X = np.load('4X_sequences.npy')  # shape: (N, 30, 30)
y = np.load('4y_labels.npy')     # shape: (N, )

# 표준 정규화
X = (X - np.mean(X)) / np.std(X)

# 3. 학습/검증 분할
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Dataset 정의
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SequenceDataset(X_train, y_train)
val_dataset = SequenceDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5. LSTM 모델 정의
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

model = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)

# 6. 손실 함수 및 옵티마이저 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 클래스 가중치: [0.908, 1.112]
class_weights = torch.tensor([0.908, 1.112], dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 저장된 모델 가중치 업로드
uploaded = files.upload()
model.load_state_dict(torch.load('lstm_4f_classifier_weighted1.112_v8.pt'))
print("저장된 모델 가중치 불러오기 완료")

# 7. 학습 루프
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # 검증
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_accuracy:.4f}")

# 8. 모델 저장
torch.save(model.state_dict(), 'lstm_4f_classifier_weighted1.112_v9.pt')
print("모델 저장 완료")

# 9. Colab에서 다시 다운로드
files.download('lstm_4f_classifier_weighted1.112_v9.pt')
