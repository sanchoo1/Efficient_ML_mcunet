import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import os
import psutil
import time
from tqdm import tqdm
from mcunet.model_zoo import build_model
import matplotlib.pyplot as plt
import math
import copy

def get_prune_ratio(layer_name):
    """
    根据层名决定剪枝比例
    """
    if 'depth_conv' in layer_name:
        return 0
    elif 'point_linear' in layer_name:
        return 0
    elif 'classifier' in layer_name:
        return 0
    elif 'conv' in layer_name:
        return 0
    else:
        return 0

def prune_conv_layer(conv_layer: nn.Conv2d, prune_ratio: float):
    """
    对一个Conv2D层做L1剪枝
    """
    with torch.no_grad():
        weight = conv_layer.weight.data
        out_channels = weight.size(0)

        l1_norms = torch.norm(weight.view(out_channels, -1), p=1, dim=1)
        num_prune = int(out_channels * prune_ratio)

        if num_prune == 0:
            mask = torch.ones(out_channels, dtype=torch.bool, device=weight.device)
            return mask
        
        _, prune_indices = torch.topk(l1_norms, num_prune, largest=False)
        mask = torch.ones(out_channels, dtype=torch.bool, device=weight.device)
        mask[prune_indices] = False

        # 把剪掉的通道对应权重置为0
        conv_layer.weight.data[~mask, :, :, :] = 0

        return mask

def prune_model(model: nn.Module):
    """
    输入模型，返回剪枝后的模型
    """
    pruned_model = copy.deepcopy(model)
    masks = {}

    for name, layer in pruned_model.named_modules():
        if isinstance(layer, nn.Conv2d):
            prune_ratio = get_prune_ratio(name)
            if prune_ratio > 0:
                print(f"Pruning {name} with ratio {prune_ratio}")
                mask = prune_conv_layer(layer, prune_ratio)
                masks[name] = mask

    return pruned_model, masks

def save_model(model: nn.Module, save_path: str):
    """
    保存模型
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


# ------------------ 配置部分 ------------------
MODEL_NAME = "mcunet-in4"
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

checkpoint_path = "/Users/daniel_feng/Desktop/MCUNet_efficient/Stage2/trained_model_mcunet-in4.pth"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    print(f"✅ Loaded pretrained weights from {checkpoint_path}")
else:
    print(f"⚠️ Warning: No pretrained model found at {checkpoint_path}, using randomly initialized model!")

model = model.to(DEVICE)
model.eval()

#------------------
# 2. 剪枝
pruned_model, masks = prune_model(model)

# 3. 保存剪枝后的模型
save_model(pruned_model, 'pruned_models/pruned_mcunet_in4.pth')