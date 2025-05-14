# Created on: 2025/5/13 by hmq_h
# Description:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 使用GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),                    # 转为Tensor
    transforms.Normalize((0.1307,), (0.3081,)) # 标准化
])

# 下载并加载数据
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # 输入通道=1，输出=32，卷积核=3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)  # 展平后是 64通道×12×12 = 9216
        self.fc2 = nn.Linear(128, 10)    # 输出是10类数字

    def forward(self, x):
        x = F.relu(self.conv1(x))        # [1,28,28] → [32,26,26]
        x = F.relu(self.conv2(x))        # [32,26,26] → [64,24,24]
        x = F.max_pool2d(x, 2)           # → [64,12,12]
        x = self.dropout1(x)
        x = torch.flatten(x, 1)          # 拉平成一维
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}')

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    accuracy = correct / len(test_loader.dataset)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
