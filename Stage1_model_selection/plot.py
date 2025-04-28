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
import matplotlib.pyplot as plt
from mcunet.model_zoo import build_model

# ------------------ 配置部分 ------------------
MODEL_NAMES = ["mcunet-in0", "mcunet-in1", "mcunet-in2", "mcunet-in3", "mcunet-in4"]
METRICS_DIR = "./"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ 解析txt文件函数 ------------------
def parse_metrics_file(file_path):
    metrics = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Average Inference Time" in line:
                metrics["Average Inference Time (ms)"] = float(line.split(":")[1].strip().split()[0])
            elif "Top1 Test Accuracy" in line:
                metrics["Top1 Test Accuracy (%)"] = float(line.split(":")[1].strip().replace('%', ''))
            elif "Single image FLOPs" in line:
                value = line.split(":")[1].strip()
                if value.endswith("M"):
                    metrics["Single image FLOPs (M)"] = float(value[:-1])
            elif "Model Parameters" in line:
                value = line.split(":")[1].strip()
                if value.endswith("K"):
                    metrics["Model Parameters (K)"] = float(value[:-1])
                elif value.endswith("M"):
                    metrics["Model Parameters (K)"] = float(value[:-1]) * 1000  # M换算成K
    return metrics

# ------------------ 收集所有模型数据 ------------------
all_metrics = {metric: {} for metric in [
    "Average Inference Time (ms)",
    "Top1 Test Accuracy (%)",
    "Single image FLOPs (M)",
    "Model Parameters (K)"
]}

for model_name in MODEL_NAMES:
    metrics_file = os.path.join(METRICS_DIR, f"{model_name}_metrics.txt")
    if os.path.exists(metrics_file):
        metrics = parse_metrics_file(metrics_file)
        for key in all_metrics.keys():
            all_metrics[key][model_name] = metrics.get(key, 0)

# ------------------ 绘制子图 ------------------
def plot_metrics(all_metrics):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    colors = ['b', 'g', 'r', 'c']

    for idx, (metric_name, values) in enumerate(all_metrics.items()):
        models = list(values.keys())
        scores = list(values.values())

        axs[idx].plot(models, scores, marker='o', linestyle='-', color=colors[idx % len(colors)], label=metric_name)
        for i, score in enumerate(scores):
            axs[idx].text(i, score, f"{score:.2f}", ha='center', va='bottom', fontsize=8)
        axs[idx].set_title(metric_name)
        axs[idx].set_xlabel("Models")
        axs[idx].set_ylabel(metric_name)
        axs[idx].grid(True)
        axs[idx].legend()

    plt.tight_layout()
    plt.savefig("all_metrics_combined.png")
    plt.show()

# ------------------ 执行绘图 ------------------
plot_metrics(all_metrics)

print("✅ All metric plots have been generated!")
