#!/usr/bin/env python3
"""
train_mcunet.py

Train MCUNet variants on Tiny ImageNet or CIFAR-100 with two-phase fine-tuning,
logging metrics to TensorBoard and CSV, and saving checkpoints & best models.
"""
import os
import time
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from mcunet.model_zoo import build_model

# —— Dataset wrappers —— #
class CIFAR100Wrapper:
    def __init__(self, data_dir="data", batch_size=128, num_workers=2):
        tf = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        tf_val = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        train_ds = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=tf)
        val_ds   = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=tf_val)
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
        self.val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

class TinyImageNetWrapper:
    def __init__(self, data_dir="data/tiny-imagenet", batch_size=128, num_workers=2):
        tf = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        tf_val = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        train_dir = os.path.join(data_dir, "train")
        val_dir   = os.path.join(data_dir, "val", "images")
        train_ds = datasets.ImageFolder(train_dir, transform=tf)
        val_ds   = datasets.ImageFolder(val_dir,   transform=tf_val)
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
        self.val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

# —— Training / Evaluation —— #
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
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
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total * 100

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * y.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return running_loss / total, correct / total * 100

# —— Main —— #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",    type=str,   required=True,
                        help="mcunet-in1|mcunet-in2|mcunet-in3|mcunet-in4")
    parser.add_argument("--dataset",     type=str,   choices=["tinyimagenet", "cifar100"], required=True)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--phase1_epochs", type=int, default=3, help="freeze backbone, train head")
    parser.add_argument("--phase2_epochs", type=int, default=15, help="fine-tune full model")
    parser.add_argument("--lr",           type=float, default=0.05)
    parser.add_argument("--momentum",     type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=2)
    parser.add_argument("--log_dir",      type=str,   default="logs")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    csv_path = os.path.join(args.log_dir, f"{args.model_id}_{args.dataset}_metrics.csv")
    with open(csv_path, "w", newline="") as cf:
        csv_writer = csv.writer(cf)
        csv_writer.writerow([
            "epoch", "phase",
            "train_loss", "train_acc",
            "val_loss",   "val_acc",
            "lr", "epoch_time"
        ])

    # Dataset
    if args.dataset == "tinyimagenet":
        data = TinyImageNetWrapper(batch_size=args.batch_size, num_workers=args.num_workers)
        num_classes = 200
    else:
        data = CIFAR100Wrapper(batch_size=args.batch_size, num_workers=args.num_workers)
        num_classes = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model, _, _ = build_model(args.model_id, pretrained=True)
    # replace head
    if hasattr(model, "classifier"):
        in_feats = model.classifier.in_features
        model.classifier = nn.Linear(in_feats, num_classes)
    else:
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    model = model.to(device)

    optimizer = SGD(model.parameters(), lr=args.lr,
                    momentum=args.momentum, weight_decay=args.weight_decay)
    total_epochs = args.phase1_epochs + args.phase2_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(1, total_epochs + 1):
        phase = "phase1" if epoch <= args.phase1_epochs else "phase2"
        # freeze or unfreeze
        if phase == "phase1":
            for p in model.parameters():
                p.requires_grad = False
            # only head params
            head = model.classifier if hasattr(model, "classifier") else model.fc
            for p in head.parameters():
                p.requires_grad = True
        elif epoch == args.phase1_epochs + 1:
            # unfreeze all
            for p in model.parameters():
                p.requires_grad = True

        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, data.train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, data.val_loader, criterion, device)
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # log
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train",  train_acc,  epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("Acc/val",    val_acc,    epoch)
        writer.add_scalar("LR",         current_lr, epoch)
        writer.add_scalar("Time/epoch", epoch_time, epoch)

        with open(csv_path, "a", newline="") as cf:
            csv.writer(cf).writerow([
                epoch, phase,
                f"{train_loss:.4f}", f"{train_acc:.2f}",
                f"{val_loss:.4f}",   f"{val_acc:.2f}",
                f"{current_lr:.6f}", f"{epoch_time:.2f}"
            ])

        # checkpoints
        ckpt_name = f"{args.model_id}_{args.dataset}_ep{epoch:02d}.pth"
        torch.save(model.state_dict(), ckpt_name)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       f"{args.model_id}_{args.dataset}_best.pth")

        scheduler.step()

        print(f"[{phase}] Epoch {epoch}/{total_epochs} "
              f"train_acc={train_acc:.2f}% val_acc={val_acc:.2f}% "
              f"lr={current_lr:.5f} time={epoch_time:.1f}s")

    # final
    final_name = f"{args.model_id}_{args.dataset}_final.pth"
    torch.save(model.state_dict(), final_name)
    print(f"Training complete. Best val_acc={best_acc:.2f}%. Saved final model to {final_name}")

if __name__ == "__main__":
    main()
