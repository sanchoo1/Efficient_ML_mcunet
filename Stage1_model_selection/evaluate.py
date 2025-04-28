import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from thop import profile, clever_format
import os
import psutil
import time
from tqdm import tqdm
from mcunet.model_zoo import build_model

# ------------------ 配置部分 ------------------
MODEL_NAME = "mcunet-in0"
SAVE_PATH = f"{MODEL_NAME}_metrics.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ 数据集 ------------------
transform_test = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

# 只取一部分测试集进行评估，比如取前1000张
subset_size = 500
subset_testset, _ = random_split(testset, [subset_size, len(testset) - subset_size])

testloader = DataLoader(subset_testset, batch_size=1, shuffle=False, num_workers=0)

# ------------------ 加载模型 ------------------
model, resolution, desc = build_model(MODEL_NAME)

if hasattr(model, 'classifier'):
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 10)
else:
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)

checkpoint_path = f"checkpoints/trained_model_{MODEL_NAME}.pth"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    print(f"✅ Loaded pretrained weights from {checkpoint_path}")
else:
    print(f"⚠️ Warning: No pretrained model found at {checkpoint_path}, using randomly initialized model!")

model = model.to(DEVICE)
model.eval()

# ------------------ 计算模型特征 ------------------
dummy_input = torch.randn(1, 3, 160, 160).to(DEVICE)
flops, params = profile(model, inputs=(dummy_input,), verbose=False)
flops, params = clever_format([flops, params], "%.3f")

# 测试 Accuracy 和平均推理时间
def evaluate(model, dataloader, device="cpu"):
    correct = 0
    total = 0
    inference_times = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            start_time = time.perf_counter()
            outputs = model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            inference_times.append((end_time - start_time) * 1000)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    avg_infer_time = sum(inference_times) / len(inference_times)

    return accuracy, avg_infer_time

# ------------------ 开始测试 ------------------
test_acc, avg_infer_time = evaluate(model, testloader, device=DEVICE)

param_size_mb = (sum(p.numel() for p in model.parameters() if p.requires_grad) * 4) / (1024 ** 2)
input_size_mb = (1 * 3 * 160 * 160 * 4) / (1024 ** 2)
forward_size_mb = input_size_mb * 2 + param_size_mb
estimated_total_size_mb = input_size_mb + forward_size_mb + param_size_mb

# ------------------ 生成文件 ------------------
with open(SAVE_PATH, "w") as f:
    f.write(f"Average Inference Time: {avg_infer_time:.2f} ms\n")
    f.write(f"Top1 Test Accuracy: {test_acc:.2f}%\n")
    f.write("="*40 + "\n")
    f.write(f"Total params: {sum(p.numel() for p in model.parameters())}\n")
    f.write(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
    f.write(f"Non-trainable params: {sum(p.numel() for p in model.parameters() if not p.requires_grad)}\n")
    f.write("="*40 + "\n")
    f.write(f"Input size (MB): {input_size_mb:.2f}\n")
    f.write(f"Forward/backward pass size (MB): {forward_size_mb:.2f}\n")
    f.write(f"Params size (MB): {param_size_mb:.2f}\n")
    f.write(f"Estimated Total Size (MB): {estimated_total_size_mb:.2f}\n")
    f.write("="*40 + "\n")
    f.write(f"Single image FLOPs: {flops}\n")
    f.write(f"Model Parameters: {params}\n")
    f.write("="*40 + "\n")

print(f"✅ Metrics saved to {SAVE_PATH}")
