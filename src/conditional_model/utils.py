import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image
from tqdm import tqdm
# Add path to the path variables
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This is `src/conditional_model`
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../.."))  # This is `src`

sys.path.append(DATA_DIR)
from data.utils import read_image_pillow, convert_to_uint8, save_image_pillow

def compute_mean_std(root, image_size=128):
    root_original = os.path.join(root, f"downsampled_images_{image_size}")
    root_blurred = os.path.join(root, f"blurred_images_{image_size}")

    image_files = os.listdir(root_original)

    transform = T.Compose([
        T.Grayscale(num_output_channels=1),  # Ensure grayscale
        T.Resize((image_size, image_size)),  # Resize
        T.ToTensor()
    ])

    sum_pixels = 0.0
    sum_squared_pixels = 0.0
    num_pixels = 0

    for img_name in tqdm(image_files, desc="Computing Mean/Std"):
        img_path = os.path.join(root_original, img_name)
        img = Image.open(img_path).convert("L")  # Convert to grayscale
        img = transform(img)  # Shape: (1, H, W)

        sum_pixels += img.sum()
        sum_squared_pixels += (img ** 2).sum()
        num_pixels += img.numel()  # Total number of pixels

    # Compute mean and std
    mean = sum_pixels / num_pixels
    std = torch.sqrt((sum_squared_pixels / num_pixels) - (mean ** 2))

    return mean.item(), std.item()

class BlurredDataset(torch.utils.data.Dataset):
    """Dataset class for loading blurred images."""
    def __init__(self, root, image_size=128):
        self.root = root
        self.image_size = image_size
        self.root_original = root + f"/downsampled_images_{image_size}"
        self.root_blurred = root + f"/blurred_images_{image_size}"
        self.original_image_files = os.listdir(self.root_original)
        self.blurred_image_files = os.listdir(self.root_blurred)
        mean, std = compute_mean_std(root, image_size)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(mean=[mean], std=[std])
        ])

    def __len__(self):
        # return len(self.original_image_files)
        return 1000

    def __getitem__(self, idx):
        image_original = read_image_pillow(os.path.join(self.root_original, self.original_image_files[idx]))
        image_blurred = read_image_pillow(os.path.join(self.root_blurred, self.blurred_image_files[idx]))
        # Add extra dimension for channel
        image_original = image_original[:, :, None]
        image_blurred = image_blurred[:, :, None]
        # Concatenate them on channel dimension
        image = np.concatenate([image_original, image_blurred], axis=2)
        image = self.transform(image)
        return image
    
if __name__ == "__main__":
    root = os.path.join(DATA_DIR, "data/")
    dataset = BlurredDataset(root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for images in dataloader:
        break