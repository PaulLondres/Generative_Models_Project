from dataclasses import dataclass
from utils import BlurredDataset
import torch
from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from accelerate import Accelerator, notebook_launcher
import numpy as np
from utils import DDPMPipeline_
from models import Encoder, UNetWithScaledBlurConditioning
import matplotlib.pyplot as plt
import lpips
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.linalg import sqrtm
from PIL import Image
import json
from torchmetrics.image.kid import KernelInceptionDistance

# Datasets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This is `src/conditional_model`
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/"))  # This is `src`
dataset = BlurredDataset(DATA_DIR, image_size=64)
# split dataset: train, val, test
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.1), int(len(dataset)*0.1)])
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

def to_uint8(tensor):
    """Convert a normalized [0,1] float tensor to uint8 format [0,255]."""
    if tensor.dtype != torch.uint8:
        return (tensor * 255).clamp(0, 255).to(torch.uint8)
    return tensor

def to_rgb(tensor):
    """Convert a 1-channel grayscale tensor to 3-channel RGB."""
    tensor = tensor.clone()
    if tensor.dtype == torch.uint8:
        return tensor.float() / 255.0  # Convert [0,255] → [0,1]
    min_val, max_val = tensor.min(), tensor.max()
    tensor = (tensor - min_val) / (max_val - min_val)  # Normalize to [0,1]
    return (tensor.repeat(1, 3, 1, 1) * 255).to(torch.uint8)  # Repeat across channel dimension

def normalize_for_metrics(tensor):
    """Ensure images are in [0,1] range before computing PSNR & SSIM."""
    tensor = tensor.clone()
    if tensor.dtype == torch.uint8:
        return tensor.float() / 255.0  # Convert [0,255] → [0,1]
    min_val, max_val = tensor.min(), tensor.max()
    # return as torch.uint8
    return (((tensor - min_val) / (max_val - min_val)) * 255).to(torch.uint8)


unet = UNet2DModel(

    sample_size=64,  # the target image resolution

    in_channels=1,  # the number of input channels, 3 for RGB images

    out_channels=1,  # the number of output channels

    layers_per_block=2,  # how many ResNet layers to use per UNet block

    block_out_channels=(128, 256, 512, 512),  # the number of output channels for each UNet block

    down_block_types=(
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
    ),

    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),

).half().to("cuda")


feature_extractor = Encoder(output_channels=1).half().to("cuda")
model = UNetWithScaledBlurConditioning(unet, feature_extractor)
# Load weights
model.load_state_dict(torch.load(f"{BASE_DIR}/model_weights_v1.pth"))
model.half().to("cuda")


noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")

pipeline = DDPMPipeline_(unet=model, 
                         scheduler=noise_scheduler, 
                         dataset=train_dataset
                         )

# images = pipeline(batch_size=1, num_inference_steps=100, output_type="pil",generator=torch.Generator(device='cpu')).images

kid_metric = KernelInceptionDistance(subset_size=50)
kid_metric_b = KernelInceptionDistance(subset_size=50)
# iterate through validation loader
psnr_values, ssim_values = [], []
# blurred
psnr_values_b, ssim_values_b = [], []
lpips_loss = lpips.LPIPS(net='alex')
kid_scores = []
kid_scores_b = []
# lpips holder
lpips_values = []
lpips_values_b = []
for i, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):  
    with torch.no_grad():
        generator = torch.Generator(device='cpu')
        image_ = data[0][0].unsqueeze(0)
        image = torch.randn(image_.shape, generator=generator, dtype=model.dtype).to("cuda")
        image_data = data[0][1].unsqueeze(0).to("cuda")
        image_cat = torch.cat([image, image_data], dim=0).half().unsqueeze(0)
        image = image.half().unsqueeze(0)
        for t in pipeline.scheduler.timesteps:
            model_output = model(image_cat, t).sample
            image = pipeline.scheduler.step(model_output, t, image, generator=generator).prev_sample
            image_cat = torch.cat([image, image_data.unsqueeze(0)], dim=1).half()

        # print(image_)
        # print(image)

        orig_uint8 = (normalize_for_metrics(image_).squeeze(0).cpu().numpy()).astype(np.uint8)
        recon_uint8 = (normalize_for_metrics(image).squeeze(0).squeeze(0).cpu().numpy()).astype(np.uint8)
        blurred_uint8 = (normalize_for_metrics(data[0][1]).squeeze(0).cpu().numpy()).astype(np.uint8)
        
        psnr_values.append(psnr(orig_uint8, recon_uint8, data_range=255))
        ssim_values.append(ssim(orig_uint8, recon_uint8, data_range=255))
        psnr_values_b.append(psnr(orig_uint8, blurred_uint8, data_range=255))
        ssim_values_b.append(ssim(orig_uint8, blurred_uint8, data_range=255))
        
        orig_tensor = to_rgb(image_).to("cpu")
        recon_tensor = to_rgb(image).to("cpu")
        blurred_tensor = to_rgb(data[0][1]).to("cpu")
        
        kid_metric.update(orig_tensor, real=True)
        kid_metric_b.update(orig_tensor, real=True)
        kid_metric.update(recon_tensor, real=False)
        kid_metric_b.update(blurred_tensor, real=False)

        # compute lpips
        lpips_loss_val = lpips_loss(orig_tensor, recon_tensor).item()
        lpips_loss_val_b = lpips_loss(orig_tensor, blurred_tensor).item()

        lpips_values.append(lpips_loss_val)
        lpips_values_b.append(lpips_loss_val_b)
        
        images = [Image.fromarray(orig_uint8), Image.fromarray(recon_uint8), Image.fromarray(blurred_uint8)]
        images = [img.convert("RGB") for img in images]

        # Create a figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns

        # Plot each image while preserving original colors
        for ax, img, title in zip(axes, images, ["Original", "Reconstructed", "Blurred"]):
            ax.imshow(np.array(img))  # Convert PIL image to NumPy array
            ax.set_title(title)
            ax.axis("off")  # Hide axes
        plt.savefig(BASE_DIR+f"/figures/validation/combined_images_rgb_{i}.png", bbox_inches="tight", dpi=300)

# Save the correctly colored image
plt.savefig(BASE_DIR+"/figures/combined_images_rgb.png", bbox_inches="tight", dpi=300)
print(f"Average PSNR: {np.mean(psnr_values):.2f}")
print(f"Average SSIM: {np.mean(ssim_values):.4f}")
kid = kid_metric.compute()[0].item()
print(f"Average KID: {kid:.4f}")
# average lpips
print(f"Average LPIPS: {np.mean(lpips_values):.4f}")
print(f"Average PSNR Blurred: {np.mean(psnr_values_b):.2f}")
print(f"Average SSIM Blurred: {np.mean(ssim_values_b):.4f}")
kid_b = kid_metric_b.compute()[0].item()
print(f"Average KID Blurred: {kid_b:.4f}")
# average lpips
print(f"Average LPIPS Blurred: {np.mean(lpips_values_b):.4f}")

dict_results = {"psnr": np.mean(psnr_values), "ssim": np.mean(ssim_values), "kid": kid, "lpips": np.mean(lpips_values)}
baseline_results = {"psnr": np.mean(psnr_values_b), "ssim": np.mean(ssim_values_b), "kid": kid_b, "lpips": np.mean(lpips_values_b)}

# Save them as json
with open(BASE_DIR+"/figures/psnr_ssim_kid.json", "w") as f:
    json.dump(dict_results, f)

with open(BASE_DIR+"/figures/psnr_ssim_kid_blurred.json", "w") as f:
    json.dump(baseline_results, f)
        



