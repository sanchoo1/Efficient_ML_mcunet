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

def visualize_weight_distribution(model, bins=256, exclude_zero=False, save_path=None):
    """
    Visualize and optionally save the weight distribution of a model.

    Args:
        model (nn.Module): The PyTorch model to visualize.
        bins (int): Number of bins for the histogram.
        exclude_zero (bool): Whether to exclude zero weights.
        save_path (str or None): If provided, save the figure to this path.
    """
    weight_data = []  # Collect weights to be visualized
    layer_names = []

    for layer_name, param_tensor in model.named_parameters():
        if param_tensor.dim() > 1:  # Only consider weight tensors, ignore biases and batchnorm params
            param_numpy = param_tensor.detach().cpu().numpy()
            flattened_numpy = param_numpy.flatten()
            weight_data.append(flattened_numpy)
            layer_names.append(layer_name)

    num_layers = len(weight_data)
    columns = 3
    rows = math.ceil(num_layers / columns)

    fig, axes = plt.subplots(rows, columns, figsize=(10, rows * 2))
    axes = axes.flatten()

    for i, (layer_weights, layer_name) in enumerate(zip(weight_data, layer_names)):
        ax = axes[i]
        if exclude_zero:
            layer_weights = layer_weights[layer_weights != 0]
        ax.hist(layer_weights, bins=bins, density=True, color='blue', alpha=0.7)
        ax.set_title(layer_name, fontsize=8)
        ax.set_xlabel('Weight Value', fontsize=6)
        ax.set_ylabel('Density', fontsize=6)
        ax.tick_params(axis='both', which='major', labelsize=6)

    for j in range(num_layers, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if needed
        plt.savefig(save_path, dpi=300)
        print(f"Saved weight distribution plot to {save_path}")
 
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

visualize_weight_distribution(
    model,
    bins=256,
    exclude_zero=False,
    save_path='weight_distribution/mcunet_in4_weights.png'
)

