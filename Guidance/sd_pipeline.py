# Original code1: https://github.com/huggingface/diffusers/blob/v0.20.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
# Original code2: https://huggingface.co/blog/stable_diffusion
import torch
import torch.nn as nn

import os
from tqdm import tqdm
from PIL import Image
from diffusers import DDIMScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionXLPipeline
from transformers import logging, CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

class StableDiffusion(nn.Module):
    def __init__(self, device, config):
        super().__init__()

        self.model_key = config.pretrain_model_id
        self.model_path = os.path.expanduser(config.pretrain_model_path)
        self.dtype = torch.float16 if config.fp16 else torch.float32
        self.device = device

        print(f"[INFO] loading {self.model_key}...")

        if self.model_path:
            self.vae = AutoencoderKL.from_pretrained(os.path.join(self.model_path, 'vae'),
                                                     variant = 'fp16').to(device)
            self.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(self.model_path, 'tokenizer'))
            if self.model_key == "stabilityai/stable-diffusion-xl-base-1.0":
                self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.model_path, subfolder="tokenizer_2")
                self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(self.model_path, subfolder="text_encoder_2").to(device)
            self.text_encoder = CLIPTextModel.from_pretrained(os.path.join(self.model_path, 'text_encoder'),
                                                              variant = 'fp16').to(device)
            self.unet = UNet2DConditionModel.from_pretrained(os.path.join(self.model_path, 'unet'),
                                                             variant = 'fp16').to(device)
            self.scheduler = DDIMScheduler.from_pretrained(os.path.join(self.model_path, 'scheduler'))
        else:
            self.vae = AutoencoderKL.from_pretrained(self.model_key, subfolder="vae").to(device)
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_key, subfolder="tokenizer")
            if self.model_key == "stabilityai/stable-diffusion-xl-base-1.0":
                self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.model_key, subfolder="tokenizer_2")
                self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(self.model_key, subfolder="text_encoder_2").to(device)
            self.text_encoder = CLIPTextModel.from_pretrained(self.model_key, subfolder="text_encoder").to(device)
            self.unet = UNet2DConditionModel.from_pretrained(self.model_key, subfolder="unet").to(device)
            self.scheduler = DDIMScheduler.from_pretrained(self.model_key, subfolder="scheduler")

        #print(dir(self.scheduler))
        print(self.scheduler.final_alpha_cumprod)
        print(f"[INFO] loaded {self.model_key}!")

    @torch.no_grad()
    def get_text_embeddings(self, prompt, negative_prompt):
        # Define tokenizers and text encoders
        if self.model_key == "stabilityai/stable-diffusion-xl-base-1.0":
            tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
            text_encoders = (
                [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
            )
            prompt = [prompt] if isinstance(prompt, str) else prompt
            prompt_2 = prompt

            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                        truncation=True, return_tensors="pt")
                prompt_embeds = text_encoder(text_input.input_ids.to(self.device), output_hidden_states=True)
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                prompt_embeds_list.append(prompt_embeds)
            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

            negative_prompt = negative_prompt or ""
            negative_prompt = len(prompt) * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = negative_prompt

            uncond_tokens = []
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif len(prompt) != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {len(prompt)}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                max_length = prompt_embeds.shape[1]
                uncond_input = self.tokenizer(negative_prompt, padding="max_length", max_length=max_length,
                                              truncation=True, return_tensors="pt")
                negative_prompt_embeds = text_encoder(uncond_input.input_ids.to(self.device), output_hidden_states=True)
                pooled_negative_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
                negative_prompt_embeds_list.append(negative_prompt_embeds)
            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=self.device)
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=self.device)
            return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, pooled_negative_prompt_embeds
        elif self.model_key == "stabilityai/stable-diffusion-2-1":
            text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                        truncation=True, return_tensors="pt")
            max_length = text_input.input_ids.shape[-1]
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
            uncond_input = self.tokenizer(negative_prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                          truncation=True, return_tensors="pt")
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            return torch.cat([uncond_embeddings, text_embeddings])

    @torch.no_grad()
    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids


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
    """
    dict = {"prompt": "cat",
            "pixel": (512, 512),
            "num_steps": 50,
            "cfg": 7.5}
    """

    @torch.no_grad()
    def get_image_latents(self, text_embeddings, height=64, width=64, num_steps=50, cfg=7.5):
        if self.model_key == "stabilityai/stable-diffusion-xl-base-1.0":
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, pooled_negative_prompt_embeds = text_embeddings
            add_time_ids = self._get_add_time_ids((height, width),
                                                  (0, 0), (height, width),
                                                  dtype=prompt_embeds.dtype,
                                                  text_encoder_projection_dim=self.text_encoder_2.config.projection_dim)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(self.device)
            batch_size = prompt_embeds.shape[0]//2
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(self.device).repeat(batch_size, 1)
            add_text_embeds = torch.cat([pooled_negative_prompt_embeds, pooled_prompt_embeds], dim=0).to(self.device)
            added_cond = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        else:
            prompt_embeds = text_embeddings
            batch_size = prompt_embeds.shape[0]//2
        latents = torch.randn((batch_size, self.unet.config.in_channels, height//8, width//8)).to(self.device)
        self.scheduler.set_timesteps(num_steps)
        for t in tqdm(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents, latents])
            #latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            if self.model_key == "stabilityai/stable-diffusion-xl-base-1.0":
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds,
                                   added_cond_kwargs=added_cond,).sample
            else:
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # perform guidance
            noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        #return latents at t=0
        return latents

    def generate_img(self, img_arguments):
        prompts = img_arguments["prompt"]
        negative_prompts = img_arguments["negative_prompt"]
        width, height = img_arguments["pixel"]
        num_steps = img_arguments["num_steps"]
        cfg = img_arguments["cfg"]

        latents = self.get_image_latents(self.get_text_embeddings(prompts, negative_prompts),
                                         height=height, width=width, num_steps=num_steps, cfg=cfg)
        images = self.decode_from_latents(latents)
        #return image ready-to-display
        return self.display_image(images)

    def train_step_sds(self, candidate, text_embeddings):
        latents = self.encode_image(candidate)
        t = int(self.num_steps*torch.rand(1, dtype=torch.float16, device=self.device))

        # get predicted noise
        noise = torch.randn_like(latents)
        noise_latents = self.scheduler.add_noise(latents, noise, t)
        noise_pred = self.unet(torch.cat([noise_latents, noise_latents]),
                               t, encoder_hidden_states=text_embeddings).sample

        # get time weighting
        w = 1
        return



    def train_step_vsd(self):
        print("No problem!")