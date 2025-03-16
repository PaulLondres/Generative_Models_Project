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

    image_files = os.listdir(root)

    transform = T.Compose([
        T.ToTensor(),  # Converts to [0, 1]
        lambda x: x * 255  # Scale back to [0, 255]
    ])

    sum_pixels = 0.0
    sum_squared_pixels = 0.0
    num_pixels = 0

    for img_name in tqdm(image_files, desc="Computing Mean/Std"):
        img_path = os.path.join(root, img_name)
        img = Image.open(img_path)  # Convert to grayscale
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
        # compute only if no dist.json file
        self.mean_org, self.std_org = compute_mean_std(self.root_original, image_size)
        self.mean_blur, self.std_blur = compute_mean_std(self.root_blurred, image_size)
        self.transform = T.Compose([
            # T.ToTensor(),  # Converts both channels simultaneously
            T.Normalize(mean=[self.mean_org, self.mean_blur], std=[self.std_org, self.std_blur])  # Normalize separately
        ])

    def __len__(self):
        return len(self.original_image_files)

    def __getitem__(self, idx):
        image_original = read_image_pillow(os.path.join(self.root_original, self.original_image_files[idx]))
        image_blurred = read_image_pillow(os.path.join(self.root_blurred, self.blurred_image_files[idx]))
        image_original = np.array(image_original, dtype=np.float32)
        image_blurred = np.array(image_blurred, dtype=np.float32)
        # Concatenate them on channel dimension
        image = np.stack([image_original, image_blurred], axis=0)
        image = torch.tensor(image)
        image = self.transform(image)
        return image
    
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Tuple, Union

import torch

from diffusers import DDPMPipeline
from diffusers import DiffusionPipeline, ImagePipelineOutput


class DDPMPipeline_(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, dataset):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.dataset = dataset

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        
        image_shape = (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size)

        image = torch.randn(image_shape, generator=generator, dtype=self.unet.dtype).to(self.device)
        
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # pick batch_size random indices
        indices = list(torch.randint(0, len(self.dataset), (batch_size,)))
        image_data = [self.dataset[i.item()] for i in indices]
        image_data = torch.stack(image_data, dim=0)[:,1,:,:].unsqueeze(1).to(self.device)  # only original image
        # change image data type to match image
        # concat image and image_data on second dimension
        image_cat = torch.cat([image, image_data], dim=1).half()
        image = image.half()
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image_cat, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample
            image_cat = torch.cat([image, image_data], dim=1)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)



if __name__ == "__main__":
    root = os.path.join(DATA_DIR, "data/")
    dataset = BlurredDataset(root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for images in dataloader:
        break