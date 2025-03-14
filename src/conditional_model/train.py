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
from tqdm import tqdm
from accelerate import Accelerator, notebook_launcher

@dataclass
class TrainingConfig:
    image_size = 64  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 4  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = f"ddpm-c-blurring-{image_size}"  # the model name locally and on the HF Hub
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This is `src/conditional_model`
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/"))  # This is `src`
dataset = BlurredDataset(DATA_DIR, image_size=config.image_size)
# split dataset: train, val, test
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.1), int(len(dataset)*0.1)])
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

model = UNet2DModel(

    sample_size=config.image_size,  # the target image resolution

    in_channels=2,  # the number of input channels, 3 for RGB images

    out_channels=1,  # the number of output channels

    layers_per_block=2,  # how many ResNet layers to use per UNet block

    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block

    down_block_types=(

        "DownBlock2D",  # a regular ResNet downsampling block

        "DownBlock2D",

        "DownBlock2D",

        "DownBlock2D",

        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention

        "DownBlock2D",

    ),

    up_block_types=(

        "UpBlock2D",  # a regular ResNet upsampling block

        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention

        "UpBlock2D",

        "UpBlock2D",

        "UpBlock2D",

        "UpBlock2D",

    ),

)

sample_image = dataset[0].unsqueeze(0)

print("Input shape:", sample_image.shape)

print("Output shape:", model(sample_image, timestep=0).sample.shape)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, )

noise = torch.randn(1, 1, TrainingConfig.image_size, TrainingConfig.image_size)

timesteps = torch.LongTensor([50])

original_image = sample_image[:, 0, :, :].unsqueeze(1)
blurred_image = sample_image[:, 1, :, :].unsqueeze(1)
noisy_image = noise_scheduler.add_noise(original_image, noise, timesteps)
input_image = torch.cat([blurred_image, noisy_image], dim=1)
noise_pred = model(input_image, timesteps).sample

loss = F.mse_loss(noise_pred, noise)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(

    optimizer=optimizer,

    num_warmup_steps=config.lr_warmup_steps,

    num_training_steps=(len(train_dataloader) * config.num_epochs),

)


def evaluate(config, epoch, pipeline):

    # Sample some images from random noise (this is the backward diffusion process).

    # The default pipeline output type is `List[PIL.Image]`

    images = pipeline(

        batch_size=config.eval_batch_size,
        

        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop

    ).images

    # Make a grid out of the images

    image_grid = make_image_grid(images, rows=2, cols=2)

    # Save the images

    test_dir = os.path.join(config.output_dir, "samples")

    os.makedirs(test_dir, exist_ok=True)

    image_grid.save(f"{test_dir}/{epoch:04d}.png")

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):

    # Initialize accelerator and tensorboard logging

    accelerator = Accelerator(

        mixed_precision=config.mixed_precision,

        gradient_accumulation_steps=config.gradient_accumulation_steps,

        log_with="tensorboard",

        project_dir=os.path.join(config.output_dir, "logs"),

    )

    if accelerator.is_main_process:

        if config.output_dir is not None:

            os.makedirs(config.output_dir, exist_ok=True)

        accelerator.init_trackers("train_example")

    # Prepare everything

    # There is no specific order to remember, you just need to unpack the

    # objects in the same order you gave them to the prepare method.

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(

        model, optimizer, train_dataloader, lr_scheduler

    )

    global_step = 0

    # Now you train the model

    for epoch in tqdm(range(config.num_epochs)):

        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)

        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):

            clean_images = batch
            original_image = clean_images[:, 0, :, :].unsqueeze(1)
            blurred_image = clean_images[:, 1, :, :].unsqueeze(1)

            # Sample noise to add to the images
            noise_shape = (clean_images.shape[0], 1, config.image_size, config.image_size)            

            noise = torch.randn(noise_shape, device=clean_images.device)

            bs = original_image.shape[0]

            # Sample a random timestep for each image

            timesteps = torch.randint(

                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,

                dtype=torch.int64

            )

            # Add noise to the clean images according to the noise magnitude at each timestep

            # (this is the forward diffusion process)

            noisy_images = noise_scheduler.add_noise(original_image, noise, timesteps)
            noisy_images = torch.cat([blurred_image, noisy_images], dim=1)

            with accelerator.accumulate(model):

                # Predict the noise residual

                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

                loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)

                if accelerator.sync_gradients:

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                lr_scheduler.step()

                optimizer.zero_grad()

            progress_bar.update(1)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}

            progress_bar.set_postfix(**logs)

            accelerator.log(logs, step=global_step)

            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model

        if accelerator.is_main_process:

            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:

                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:

                pipeline.save_pretrained(config.output_dir)


args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)