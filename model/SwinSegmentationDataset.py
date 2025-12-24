import torch
import numpy as np

from torch.utils.data import Dataset
import cv2
import os

class SwinSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_size=(512, 1024)):
        """
        root_dir: base dataset directory containing train/val/test subdirs
        split: one of 'train', 'val', 'test'
        transform: optional torchvision transform for image
        target_size: size to resize images and labels to
        """
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.label_dir = os.path.join(root_dir, split, 'labels')
        self.image_paths = sorted(os.listdir(self.image_dir))
        self.label_paths = sorted(os.listdir(self.label_dir))
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # --- Load input image (grayscale or RGB) ---
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        # --- Load label mask ---
        label_path = os.path.join(self.label_dir, self.label_paths[idx])
        label = cv2.imread(label_path, cv2.IMREAD_COLOR)
        label = cv2.resize(label, self.target_size)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        # --- Convert RGB label to class indices ---
        class_mask = np.full((label.shape[0], label.shape[1]), 2, dtype=np.uint8)
        green_mask = np.all(label == [0, 255, 0], axis=-1)
        red_mask = np.all(label == [255, 0, 0], axis=-1)
        class_mask[green_mask] = 0
        class_mask[red_mask] = 1

        label_tensor = torch.from_numpy(class_mask).long()
        return image, label_tensor