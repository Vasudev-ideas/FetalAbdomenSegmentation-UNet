import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

IMG_SIZE = 256

class UltrasoundDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        
        # Only take image files
        self.files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
        self.files.sort()   # IMPORTANT for consistency

        print(f"Total samples: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        base = os.path.splitext(file)[0]

        img_path = os.path.join(self.img_dir, file)
        mask_path = os.path.join(self.mask_dir, base + ".npy")

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = np.load(mask_path)

        # DEBUG: check mask values
        # print("Mask unique:", np.unique(mask))

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE),
                          interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0

        img = torch.from_numpy(img).unsqueeze(0)      # (1, H, W)
        mask = torch.from_numpy(mask).long()          # (H, W)

        return img, mask