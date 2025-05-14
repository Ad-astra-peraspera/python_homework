# Created on: 2025/5/14 by hmq_h
# Description:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 简单的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3)
        self.fc = nn.Linear(26 * 26 * 16, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 加载 MNIST 数据集
transform = transforms.ToTensor()
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# 初始化模型与优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

# 训练一个 epoch（快速测试用）
model.train()
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    preds = model(images)
    loss = loss_fn(preds, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item():.4f}")
    break  # 只训练一批就退出，测试用