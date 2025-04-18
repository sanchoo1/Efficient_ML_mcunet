# eval_mcunet_cifar10.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from mcunet.model_zoo import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pick a model (you can change this)
net_id = "mcunet-in2"
model, image_size, desc = build_model(net_id=net_id, pretrained=True)
model = model.to(device)
model.eval()

# Adjust CIFAR-10 input to match expected model input size
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Evaluate model on CIFAR-10
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"✅ CIFAR-10 accuracy of {net_id}: {accuracy:.2f}% (Image resized to {image_size}×{image_size})")
