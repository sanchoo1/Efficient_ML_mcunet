import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from mcunet.model_zoo import build_model
from torch.utils.data import DataLoader
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net_ids = ["mcunet-in0", "mcunet-in1", "mcunet-in2", "mcunet-in3", "mcunet-in4"]

# CIFAR-10 数据加载 & 预处理
transform = transforms.Compose([
    transforms.Resize((96, 96)),  # 统一输入大小
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader  = DataLoader(testset,  batch_size=64, shuffle=False, num_workers=2)

print(f"{'Model':<12}{'Acc (%)':>10}{'Time(s)':>10}")
print("-" * 34)

for net_id in net_ids:
    # 加载 ImageNet 预训练模型
    model, image_size, desc = build_model(net_id=net_id, pretrained=True)
    
    # 替换输出层
    model.classifier = nn.Linear(model.classifier.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start = time.time()

    # === 微调训练 1 个 epoch ===
    model.train()
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    
    # === 测试 ===
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    end = time.time()
    acc = 100 * correct / total
    print(f"{net_id:<12}{acc:>10.2f}{end - start:>10.2f}")
