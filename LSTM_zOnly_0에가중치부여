from google.colab import files

uploaded = files.upload()

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from google.colab import files

# �곗씠�� 濡쒕뱶
X = np.load("X_zonly_sequences.npy")
y = np.load("y_zonly_labels.npy")

# 而ㅼ뒪�� Dataset �뺤쓽
class ZOnlySequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# LSTM 紐⑤뜽 �뺤쓽
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# �곗씠�곗뀑 以�鍮�
dataset = ZOnlySequenceDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# �대옒�� 媛�以묒튂 怨꾩궛 (�대옒�� 0�� �� �믪� 媛�以묒튂)
total = len(y)
class_weights = torch.tensor([
    total / (2 * np.sum(y == 0)),
    total / (2 * np.sum(y == 1))
], dtype=torch.float32)

# 紐⑤뜽 �ㅼ젙
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(input_size=1).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 湲곗〈 紐⑤뜽 �낅줈��
uploaded = files.upload()
model.load_state_dict(torch.load("lstm_zOnly_0媛�以묒튂_v6.pt"))

# �숈뒿 猷⑦봽
EPOCHS = 30
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
        running_loss += loss.item() * inputs.size(0)
    
    # 寃�利� �뺥솗�� 怨꾩궛
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader.dataset):.4f}, Val Acc: {val_acc:.4f}")

# 紐⑤뜽 ���� 諛� �ㅼ슫濡쒕뱶
torch.save(model.state_dict(), 
           'lstm_zOnly_0媛�以묒튂_v7.pt')
print("紐⑤뜽 ���� �꾨즺")
files.download('lstm_zOnly_0媛�以묒튂_v7.pt')
