import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.SwinSegmentationProposedModel import SwinSegmentationProposedModel
from model.SwinSegmentationDataset import SwinSegmentationDataset
import random
import fcntl
import matplotlib.pyplot as plt
import time
import sys


def pixel_accuracy(outputs, masks, num_classes, ignore_class=None):
    preds = torch.argmax(outputs, dim=1)  # (B, H, W)
    correct_pixels = (preds == masks)

    # Mask out ignored class for total accuracy
    if ignore_class is not None:
        valid_mask = (masks != ignore_class)
        total_correct = (correct_pixels & valid_mask).sum().item()
        total_pixels = valid_mask.sum().item()
    else:
        total_correct = correct_pixels.sum().item()
        total_pixels = correct_pixels.numel()

    overall_acc = total_correct / total_pixels if total_pixels > 0 else float('nan')

    # Per-class accuracy
    per_class_acc = []
    for c in range(num_classes):
        class_mask = (masks == c)
        total = class_mask.sum().item()
        if total == 0:
            acc = float('nan')
        else:
            acc = (correct_pixels & class_mask).sum().item() / total
        per_class_acc.append(acc)

    return overall_acc, per_class_acc

start_time = time.time()
num_epochs = 10



def train_swin():
    num_classes = 3  # C0=green, C1=red, C2=unclassified
    ignore_class = 2

    # Edit to corresponding images directory
    # e.g. Final Dataset Images/train/images and Final Dataset Images/train/labels
    train_dataset = SwinSegmentationDataset(root_dir='Final Dataset Images', split='train')
    val_dataset   = SwinSegmentationDataset(root_dir='Final Dataset Images', split='val')

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_class)  # Can ignore unclassified (C2) in loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = SwinSegmentationProposedModel((1024,512), 3, mlp_ratio=4, embed_dim=96, 
                                      patch_size=4, window_size=8, depths=[2,2,6,2]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_accs = []
    val_accs = []


    for epoch in range(num_epochs):
        # ---------- Training ----------
        model.train()
        train_loss = 0.0
        train_total_acc = 0.0
        train_acc_sum = torch.zeros(num_classes)

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            overall_acc, per_class_acc = pixel_accuracy(outputs, masks, num_classes, ignore_class=ignore_class)
            train_total_acc += overall_acc
            for i, acc in enumerate(per_class_acc):
                if not torch.isnan(torch.tensor(acc)):
                    train_acc_sum[i] += acc

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        avg_train_total_acc = train_total_acc / len(train_loader)
        avg_train_acc_per_class = (train_acc_sum / len(train_loader)).tolist()

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        val_total_acc = 0.0
        val_acc_sum = torch.zeros(num_classes)

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                overall_acc, per_class_acc = pixel_accuracy(outputs, masks, num_classes, ignore_class=ignore_class)
                val_total_acc += overall_acc
                for i, acc in enumerate(per_class_acc):
                    if not torch.isnan(torch.tensor(acc)):
                        val_acc_sum[i] += acc

        avg_val_loss = val_loss / len(val_loader)
        avg_val_total_acc = val_total_acc / len(val_loader)
        avg_val_acc_per_class = (val_acc_sum / len(val_loader)).tolist()

        # ---------- Print metrics ----------
        train_acc_str = ", ".join([f"C{i}: {acc*100:.2f}%" for i, acc in enumerate(avg_train_acc_per_class)])
        val_acc_str   = ", ".join([f"C{i}: {acc*100:.2f}%" for i, acc in enumerate(avg_val_acc_per_class)])

        train_accs.append(avg_train_total_acc)
        val_accs.append(avg_val_total_acc)
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Total Acc*: {avg_train_total_acc*100:.2f}% | {train_acc_str} | "
              f"Val Loss: {avg_val_loss:.4f}, Total Acc*: {avg_val_total_acc*100:.2f}% | {val_acc_str}")

    return model, train_accs, val_accs

def main():
    # Save the model
    output = train_swin()

    epochs = list(range(1, num_epochs + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, output[1], marker='o', label='Train Accuracy')
    plt.plot(epochs, output[2], marker='s', label='Validation Accuracy')

    plt.title('Train vs Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Save as PNG with high resolution
    plt.savefig("wbseg-swin.png", dpi=300)

    torch.save(output[0].state_dict(), 'wbseg-swin.pth')



if __name__ == "__main__":
    main()