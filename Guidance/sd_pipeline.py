# Original code: https://huggingface.co/blog/stable_diffusion
import torch
import torch.nn as nn
import imageio
import os
from tqdm import tqdm
from PIL import Image
from diffusers import DDIMScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
from transformers import logging, CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

class StableDiffusion(nn.Module):
    def __init__(self, device, config):
        super().__init__()
        self.model_key = config['pretrain_model_id']
        self.device = device
        self.dtype = torch.float16
        print(f"[INFO] loading {self.model_key}...")
        try:
            self.model_path = os.path.expanduser(config['pretrain_model_path'])
            self.vae = AutoencoderKL.from_pretrained(os.path.join(self.model_path, 'vae'), variant = 'fp16', torch_dtype=self.dtype).to(device)
            self.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(self.model_path, 'tokenizer'), torch_dtype=self.dtype)
            self.text_encoder = CLIPTextModel.from_pretrained(os.path.join(self.model_path, 'text_encoder'), variant = 'fp16', torch_dtype=self.dtype).to(device)
            self.unet = UNet2DConditionModel.from_pretrained(os.path.join(self.model_path, 'unet'), variant = 'fp16', torch_dtype=self.dtype).to(device)
            self.scheduler = DDIMScheduler.from_pretrained(os.path.join(self.model_path, 'scheduler'), torch_dtype=self.dtype)    
        except:
            self.vae = AutoencoderKL.from_pretrained(self.model_key, subfolder=" vae", torch_dtype=self.dtype).to(device)
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_key, subfolder="tokenizer", torch_dtype=self.dtype)
            self.text_encoder = CLIPTextModel.from_pretrained(self.model_key,  subfolder="text_encoder", torch_dtype=self.dtype).to(device)
            self.unet = UNet2DConditionModel.from_pretrained(self.model_key, subfolder="unet", torch_dtype=self.dtype).to(device)
            self.scheduler = DDIMScheduler.from_pretrained(self.model_key, subfolder="scheduler", torch_dtype=self.dtype)

        #print(dir(self.scheduler))

    def get_text_embeddings(self, prompt, negative_prompt):
            text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                        truncation=True, return_tensors="pt")
            max_length = text_input.input_ids.shape[-1]
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
            uncond_input = self.tokenizer(negative_prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                          truncation=True, return_tensors="pt")
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            return torch.cat([uncond_embeddings, text_embeddings]).detach()
    
    def decode_from_latents(self, latents):
        image_latents = 1 / 0.18215 * latents
        return self.vae.decode(image_latents).sample
    
    def encode_image(self, image):
        # images: (Batch, channels, height, width) -> latents
        if len(image.shape) < 4:
            input_img = image.unsqueeze(0)
        latent = self.vae.encode(input_img * 2 - 1)
        return 0.18215 * latent.latent_dist.sample()

    def display_image(self, images):
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        #return sequence of denoise images
        return pil_images

    @torch.no_grad()
    def get_image_latents(self, text_embeddings, batch_size=1, height=64, width=64, num_steps=50, cfg=7.5):
        latents = torch.randn((batch_size, self.unet.config.in_channels, height//8, width//8)).to(self.device)
        self.scheduler.set_timesteps(num_steps)
        for t in tqdm(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents, latents])
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        return latents

    def generate_img(self, img_arguments):
        prompts = img_arguments["prompt"]
        negative_prompts = img_arguments["negative_prompt"]
        width, height = img_arguments["pixel"]
        num_steps = 100
        cfg = img_arguments["cfg"]
        batch_size = img_arguments["batch_size"]
        latents = self.get_image_latents(text_embeddings=self.get_text_embeddings(prompts, negative_prompts), batch_size=batch_size, height=height, width=width, num_steps=num_steps, cfg=cfg)
        images = self.decode_from_latents(latents)
        #return image ready-to-display
        return self.display_image(images)
    