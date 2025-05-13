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
MODEL_NAME = 'hard'
MODEL_PATH   = 'content/mcunet-in4_2_cifar10.pth'

base = os.path.splitext(os.path.basename(MODEL_PATH))[0]
SAVE_PATH = f"{base}_evaluation.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ 数据集 ------------------
class CIFAR:
    """
    CIFAR-10 DataLoader wrapper for MCUNet (160×160 input, finetune on ImageNet-pretrained weights).
    """
    def __init__(self, batch_size=128, num_workers=0):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_loader, self.test_loader = self.prepare_data()

    def prepare_data(self):
        transform_train = transforms.Compose([
            transforms.Resize((160, 160)),              
            transforms.RandomCrop(160, padding=16),     
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(              
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((160, 160)),      
            transforms.ToTensor(),
            transforms.Normalize( 
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.batch_size,   
            shuffle=False,
            num_workers=self.num_workers,
        )

        return trainloader, testloader

dataloader = {}
cifar = CIFAR()
dataloader['train'], dataloader['test'] = cifar.train_loader, cifar.test_loader

def evaluate_with_inftime(model: nn.Module, test_dataloader: DataLoader, device="cuda") -> tuple[float, float]:
    """
    Inference function to evaluate the model on a test dataset using the provided dataloader,
    with tqdm progress bar for visualization. It also measures average inference time per batch.

    Args:
        model: The neural network model to be used for inference.
        test_dataloader: The dataloader for the test data.
        device: Device to use ('cuda' or 'cpu').

    Returns:
        Tuple of:
        - accuracy (float): Top-1 accuracy on the test set (%)
        - avg_inference_time (float): Average inference time per batch (ms)
    """
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    inference_times = []

    num_batches = len(test_dataloader)
    half_batches = num_batches // 2

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(test_dataloader, desc="Evaluating", leave=False)):
            if i >= half_batches:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            # Start timing
            start_time = time.perf_counter()
            outputs = model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()  # 等待GPU完成任务
            end_time = time.perf_counter()

            inference_times.append((end_time - start_time) * 1000)  # 转为毫秒

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    avg_infer_time = sum(inference_times) / total

    return accuracy, avg_infer_time

# ------------------ 加载模型 ------------------
def load_pretrained_model(model_name: str, weight_path: str, num_classes=10, device=DEVICE):

    model, resolution, desc = build_model(model_name)


    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Cannot find final classification layer in model")


    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model

if MODEL_NAME == 'hard':
    model = torch.load("hardpruned_.pth", map_location=DEVICE, weights_only=False)
    model.eval()
    SAVE_PATH = 'hard_evaluation.txt'
else:
    model = load_pretrained_model(MODEL_NAME, MODEL_PATH)
print("Model load successful")

# ------------------ 计算模型特征 ------------------
dummy_input = torch.randn(1, 3, 160, 160).to(DEVICE)
flops, params = profile(model, inputs=(dummy_input,), verbose=False)
flops, params = clever_format([flops, params], "%.3f")

# ------------------ 开始测试 ------------------
test_acc, avg_infer_time = evaluate_with_inftime(model, dataloader['test'], device=DEVICE)

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

print(f"Metrics saved to {SAVE_PATH}")