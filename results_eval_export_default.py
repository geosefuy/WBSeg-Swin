import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.SwinSegmentationProposedModel import SwinSegmentationProposedModel
from model.SwinSegmentationDataset import SwinSegmentationDataset
import os

# Setup device, model, dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
dataset_eval = 'test'
model_name = 'default'

# Model must match checkpoint num_classes=3
model = SwinSegmentationProposedModel(
    img_size=(1024,512),
    num_classes=3,   # match checkpoint
    mlp_ratio=4,
    embed_dim=96,
    patch_size=4,
    window_size=8,
    depths=[2,2,6,2]
)
model = model.to(device)

# Load checkpoint
state_dict = torch.load(
    "wbseg-swin.pth",
    weights_only=True,
    map_location=device
)
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)

model.eval()

# Load dataset
val_dataset = SwinSegmentationDataset(root_dir='Final Dataset Images', split=dataset_eval)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

os.makedirs("results/error_maps", exist_ok=True)

# Metric functions
def dice_score(y_true, y_pred, class_id):
    y_true_c = (y_true == class_id).astype(np.uint8)
    y_pred_c = (y_pred == class_id).astype(np.uint8)
    intersection = np.sum(y_true_c * y_pred_c)
    denom = np.sum(y_true_c) + np.sum(y_pred_c)
    return 1.0 if denom == 0 else 2.0 * intersection / denom

def iou_score(y_true, y_pred, class_id):
    y_true_c = (y_true == class_id).astype(np.uint8)
    y_pred_c = (y_pred == class_id).astype(np.uint8)
    intersection = np.sum(y_true_c * y_pred_c)
    union = np.sum(y_true_c) + np.sum(y_pred_c) - intersection
    return 1.0 if union == 0 else intersection / union

def false_negative_rate(y_true, y_pred, class_id):
    y_true_c = (y_true == class_id).astype(np.uint8)
    y_pred_c = (y_pred == class_id).astype(np.uint8)
    FN = np.sum((y_true_c == 1) & (y_pred_c == 0))
    TP = np.sum((y_true_c == 1) & (y_pred_c == 1))
    return 0.0 if (FN + TP) == 0 else FN / (FN + TP)

def overall_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def per_class_accuracy(y_true, y_pred, class_id):
    mask = y_true == class_id
    if np.sum(mask) == 0:
        return 1.0
    return np.mean(y_true[mask] == y_pred[mask])

# Collect predictions
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)               # [B,3,H,W]
        outputs_2cls = outputs[:, :2, :, :]   # slice only trained classes
        preds = torch.argmax(outputs_2cls, dim=1)  # [B,H,W]

        # Flatten and mask class 2 in labels
        preds_flat = preds.view(-1)
        labels_flat = labels.view(-1)
        mask = labels_flat != 2
        preds_flat = preds_flat[mask]
        labels_flat = labels_flat[mask]

        all_preds.append(preds_flat.cpu().numpy())
        all_labels.append(labels_flat.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

os.makedirs("results/exports", exist_ok=True)

# Save flattened arrays
np.save("results/exports/all_labels_"+ model_name + "_" + dataset_eval +".npy", all_labels)
np.save("results/exports/all_preds_"+ model_name + "_" + dataset_eval +".npy", all_preds)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds, labels=[0,1])
plt.figure(figsize=(6,5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["No Metastasis", "Metastasis"],
    yticklabels=["No Metastasis", "Metastasis"]
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix_" + model_name + "_" + dataset_eval + ".png", dpi=300)
plt.close()

# Compute DSC, IoU, FN rate
metrics = {}
for cls in [0, 1]:
    dsc = dice_score(all_labels, all_preds, cls)
    iou = iou_score(all_labels, all_preds, cls)
    fnr = false_negative_rate(all_labels, all_preds, cls)
    acc = per_class_accuracy(all_labels, all_preds, cls)
    metrics[cls] = {"DSC": dsc, "IoU": iou, "FN_rate": fnr, "Acc": acc}

# Mean metrics (classes 0 & 1)
mean_dsc = np.mean([metrics[c]["DSC"] for c in [0, 1]])
mean_iou = np.mean([metrics[c]["IoU"] for c in [0, 1]])
mean_acc = overall_accuracy(all_labels, all_preds)

print("\nMetrics Summary (Ignoring Class 2)")
for c in [0, 1]:
    name = "No Metastasis" if c == 0 else "Metastasis"
    print(f"Class {c} ({name}) ? DSC: {metrics[c]['DSC']:.4f}, IoU: {metrics[c]['IoU']:.4f}, "
          f"FN Rate: {metrics[c]['FN_rate']:.4f}, Acc: {metrics[c]['Acc']:.4f}")

print(f"\nMean DSC: {mean_dsc:.4f}")
print(f"Mean IoU: {mean_iou:.4f}")
print(f"Overall Pixel Accuracy (classes 0 & 1 only): {mean_acc:.4f}")

# Save error heatmaps
def save_error_map(image, label, pred, idx):
    image_np = image.permute(1,2,0).cpu().numpy()
    label_np = label.cpu().numpy()
    pred_np = pred.cpu().numpy()

    mask = label_np != 2
    error_map = (label_np != pred_np) & mask

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Input Image")
    plt.imshow(image_np)

    plt.subplot(1,3,2)
    plt.title("Ground Truth")
    plt.imshow(label_np, cmap="tab20")

    plt.subplot(1,3,3)
    plt.title("Error Heatmap")
    plt.imshow(error_map, cmap="Reds", alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"results/error_maps/sample_{idx:04d}.png", dpi=300)
    plt.close()

dataset_height, dataset_width = val_dataset[0][0].shape[1:]  # (C,H,W)

# Initialize accumulators
error_accum_all = np.zeros((dataset_height, dataset_width), dtype=np.uint32)
error_accum_0 = np.zeros((dataset_height, dataset_width), dtype=np.uint32)
error_accum_1 = np.zeros((dataset_height, dataset_width), dtype=np.uint32)

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        outputs_2cls = outputs[:, :2, :, :]
        preds = torch.argmax(outputs_2cls, dim=1)

        for b in range(images.size(0)):
            label_np = labels[b].cpu().numpy()
            pred_np = preds[b].cpu().numpy()

            # Ignore class 2
            mask_valid = label_np != 2
            error_map_all = (label_np != pred_np) & mask_valid
            error_accum_all += error_map_all.astype(np.uint32)

            # Class 0
            mask0 = label_np == 0
            error_map0 = (label_np != pred_np) & mask0
            error_accum_0 += error_map0.astype(np.uint32)

            # Class 1
            mask1 = label_np == 1
            error_map1 = (label_np != pred_np) & mask1
            error_accum_1 += error_map1.astype(np.uint32)

# Normalize for visualization
error_accum_all_norm = error_accum_all / error_accum_all.max()
error_accum_0_norm = error_accum_0 / error_accum_0.max()
error_accum_1_norm = error_accum_1 / error_accum_1.max()

np.save("results/exports/error_accum_all_" + model_name + "_" + dataset_eval + ".npy", error_accum_all)
np.save("results/exports/error_accum_0_" + model_name + "_" + dataset_eval + ".npy", error_accum_0)
np.save("results/exports/error_accum_1_" + model_name + "_" + dataset_eval + ".npy", error_accum_1)

np.save("results/exports/error_accum_all_norm_" + model_name + "_" + dataset_eval + ".npy", error_accum_all_norm)
np.save("results/exports/error_accum_0_norm_" + model_name + "_" + dataset_eval + ".npy", error_accum_0_norm)
np.save("results/exports/error_accum_1_norm_" + model_name + "_" + dataset_eval + ".npy", error_accum_1_norm)

# Combined error heatmap
plt.figure(figsize=(10,10))
plt.title("Error Heatmap - Combined Classes 0 & 1")
plt.imshow(error_accum_all_norm, cmap="Greys", alpha=0.7)
plt.colorbar(label="Error Frequency")
plt.tight_layout()
plt.savefig("results/error_maps/error_map_" + model_name + "_" + dataset_eval + "_combined.png", dpi=300)
plt.close()

# Class 0 heatmap
plt.figure(figsize=(10,10))
plt.title("Error Heatmap - No Metastasis")
plt.imshow(error_accum_0_norm, cmap="Greys", alpha=0.7)
plt.colorbar(label="Error Frequency")
plt.tight_layout()
plt.savefig("results/error_maps/error_map_" + model_name + "_" + dataset_eval + "_class0.png", dpi=300)
plt.close()

# Class 1 heatmap
plt.figure(figsize=(10,10))
plt.title("Error Heatmap - Metastasis")
plt.imshow(error_accum_1_norm, cmap="Greys", alpha=0.7)
plt.colorbar(label="Error Frequency")
plt.tight_layout()
plt.savefig("results/error_maps/error_map_" + model_name + "_" + dataset_eval + "_class1.png", dpi=300)
plt.close()