#!/usr/bin/env python3
"""
train_mcunet.py

Train multiple MCUNet variants on Tiny ImageNet and/or CIFAR-100 using all available GPUs,
with two-phase fine-tuning, logging to TensorBoard/CSV, and saving only best and final checkpoints.
"""
import os
import sys
import time
import csv
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from mcunet.model_zoo import build_model

# Mean/Std for CIFAR-100
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

class CIFAR100Wrapper:
    def __init__(self, resolution, data_dir="data", batch_size=128, num_workers=4):
        tf_train = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomCrop(resolution, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
        tf_val = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
        train_ds = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=tf_train)
        val_ds   = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=tf_val)
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

class TinyImageNetWrapper:
    def __init__(self, resolution, data_dir="data/tiny-imagenet", batch_size=128, num_workers=4):
        # 准备数据目录
        parent = os.path.dirname(data_dir)
        if not os.path.isdir(data_dir):
            print(f"[Info] Downloading and extracting Tiny ImageNet to {data_dir}")
            os.makedirs(parent, exist_ok=True)
            zip_path = os.path.join(parent, "tiny-imagenet-200.zip")
            os.system(f"wget -q http://cs231n.stanford.edu/tiny-imagenet-200.zip -O {zip_path}")
            os.system(f"unzip -q {zip_path} -d {parent}")
            os.system(f"mv {os.path.join(parent, 'tiny-imagenet-200')} {data_dir}")
        # 重组验证集目录
        val_dir = os.path.join(data_dir, "val")
        images_dir = os.path.join(val_dir, "images")
        if os.path.isdir(val_dir) and not os.listdir(images_dir if os.path.isdir(images_dir) else ""):
            print(f"[Info] Restructuring validation images in {val_dir}")
            ann = os.path.join(val_dir, "val_annotations.txt")
            # 创建 images 子目录
            os.makedirs(images_dir, exist_ok=True)
            # 根据注释文件移动图片
            os.system(f"cd {val_dir} && awk '{{print $2}}' val_annotations.txt | sort -u | xargs -I{{}} mkdir -p images/{{}}")
            os.system(f"cd {val_dir} && awk '{{print $1 \"	\" $2}}' val_annotations.txt | while IFS=\"	\" read img cls; do mv images/$img images/$cls/; done")
        # 数据增强与预处理
        tf_train = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomCrop(resolution, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        tf_val = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        # 构建 DataLoader
        train_dir = os.path.join(data_dir, "train")
        train_ds = datasets.ImageFolder(train_dir, transform=tf_train)
        val_ds   = datasets.ImageFolder(images_dir, transform=tf_val)
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Training and evaluation

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss/total, correct/total*100


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return running_loss/total, correct/total*100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ids", type=str, default="mcunet-in1,mcunet-in2,mcunet-in3,mcunet-in4",
                        help="Comma-separated MCUNet model IDs to train")
    parser.add_argument("--datasets", type=str, default="tinyimagenet,cifar100",
                        help="Comma-separated datasets: tinyimagenet,cifar100")
    parser.add_argument("--phase1_epochs", type=int, default=3,
                        help="Epochs for head-only training")
    parser.add_argument("--phase2_epochs", type=int, default=15,
                        help="Epochs for full fine-tuning")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr_head", type=float, default=0.01, help="Learning rate for head-only phase")
    parser.add_argument("--lr_finetune", type=float, default=0.05, help="Learning rate for full fine-tune phase")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_root", type=str, default="logs", help="Root directory for logs/checkpoints")
    args, unknown = parser.parse_known_args()  # ignore unknown args from Jupyter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ngpu = torch.cuda.device_count()
    print(f"Using device {device}, GPUs available: {ngpu}")

    model_ids = [m.strip() for m in args.model_ids.split(",")]
    dataset_names = [d.strip() for d in args.datasets.split(",")]

    for model_id in model_ids:
        for ds_name in dataset_names:
            # prepare log dirs
            combo = f"{model_id}_{ds_name}"
            log_dir = os.path.join(args.log_root, combo)
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)
            csv_path = os.path.join(log_dir, f"metrics.csv")
            with open(csv_path, "w", newline="") as cf:
                csv.writer(cf).writerow([
                    "epoch", "phase",
                    "train_loss", "train_acc",
                    "val_loss",   "val_acc",
                    "lr", "time"
                ])

            # build model
            model, resolution, _ = build_model(model_id, pretrained=True)
            num_classes = 200 if ds_name == "tinyimagenet" else 100
            # replace head
            if hasattr(model, 'classifier'):
                in_feats = model.classifier.in_features
                model.classifier = nn.Linear(in_feats, num_classes)
            else:
                in_feats = model.fc.in_features
                model.fc = nn.Linear(in_feats, num_classes)
            model = model.to(device)
            if ngpu>1:
                model = nn.DataParallel(model)

            # dataset
            if ds_name == "tinyimagenet":
                data = TinyImageNetWrapper(resolution, batch_size=args.batch_size, num_workers=args.num_workers)
            else:
                data = CIFAR100Wrapper(resolution, data_dir="data", batch_size=args.batch_size, num_workers=args.num_workers)

            # criterion, optimizer, scheduler
            criterion = nn.CrossEntropyLoss()
            # Phase1: head-only
            optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr_head, momentum=args.momentum, weight_decay=args.weight_decay)
            # Phase2 scheduler
            total_epochs = args.phase1_epochs + args.phase2_epochs
            scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

            best_acc = 0.0
            for epoch in range(1, total_epochs+1):
                phase = "phase1" if epoch<=args.phase1_epochs else "phase2"
                # freeze/unfreeze
                if phase=="phase1":
                    for p in model.parameters(): p.requires_grad=False
                    # only head
                    head = model.module.classifier if isinstance(model, nn.DataParallel) else model.classifier
                    for p in head.parameters(): p.requires_grad=True
                elif epoch==args.phase1_epochs+1:
                    for p in model.parameters(): p.requires_grad=True
                    # reset optimizer for full model
                    optimizer = SGD(model.parameters(), lr=args.lr_finetune,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
                    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

                start = time.time()
                train_loss, train_acc = train_one_epoch(model, data.train_loader, criterion, optimizer, device)
                val_loss, val_acc     = evaluate(model, data.val_loader, criterion, device)
                epoch_time = time.time()-start
                lr = optimizer.param_groups[0]['lr']

                # log
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Acc/train', train_acc, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Acc/val', val_acc, epoch)
                writer.add_scalar('LR', lr, epoch)
                writer.add_scalar('Time', epoch_time, epoch)
                with open(csv_path, 'a', newline='') as cf:
                    csv.writer(cf).writerow([epoch, phase,
                                               f"{train_loss:.4f}", f"{train_acc:.2f}",
                                               f"{val_loss:.4f}", f"{val_acc:.2f}",
                                               f"{lr:.6f}", f"{epoch_time:.1f}"])

                print(f"[{combo}][{phase}] Epoch {epoch}/{total_epochs} "
                      f"train_acc={train_acc:.2f}% val_acc={val_acc:.2f}% "
                      f"lr={lr:.5f} time={epoch_time:.1f}s")

                # best checkpoint
                if val_acc>best_acc:
                    best_acc=val_acc
                    torch.save(model.state_dict(), os.path.join(log_dir, f"{combo}_best.pth"))
                scheduler.step()

            # final checkpoint
            torch.save(model.state_dict(), os.path.join(log_dir, f"{combo}_final.pth"))
            print(f"[{combo}] Done. Best val_acc={best_acc:.2f}%\n")

if __name__ == "__main__":
    main()
