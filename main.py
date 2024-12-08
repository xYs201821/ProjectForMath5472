import imageio
import os

from Utility.miscellanea import parse_yaml, DecayWeightScheduler
import torch
from torch import nn
from tqdm import tqdm
from Guidance import StableDiffusion
from diffusers import UNet2DConditionModel
torch.set_default_dtype(torch.float16)

class lora(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=32):
        super().__init__()
        self.alpha = alpha
        self.rank = rank
        
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.scaling = self.alpha / self.rank
        
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        
    def forward(self, x):
        return self.lora_up(self.lora_down(x) * self.scaling)

def get_random_timestep(low_t, high_t, size_t=(1,)):
    return torch.randint(low_t, high_t, size_t)

@torch.no_grad()
def sds_sampling(model, config, de_renderer=1.):
    # initiate a candiate as latents
    scheduler = DecayWeightScheduler()
    height, width = config["pixel"]
    cfg = config["cfg"]
    text_embeddings = model.get_text_embeddings(config["prompt"], config["negative_prompt"])
    latents = torch.randn((config['batch_size'], model.unet.config.in_channels, height//8, width//8)).to(model.device)
    latents.requires_grad = True
    particles = [latents]
    optimizer = torch.optim.AdamW(particles, lr=1e-2)
    train_steps = 1000
    model.scheduler.set_timesteps(1000)
    list_lr = []
    weights = torch.sqrt(torch.cumprod(model.scheduler.alphas, dim=0)).to(model.device)    
    weights = torch.cumprod(model.scheduler.alphas, dim=0).to(model.device)
    #weights = (1 - model.scheduler.alphas).to(model.device)
    for itr in tqdm(range(train_steps)):
        if itr % 100 == 0:
            images = model.decode_from_latents(latents)
            imageio.imwrite(f"Test/{itr}.png",model.display_image(images)[0])
        lr = scheduler.get_decay_rate(itr)
        list_lr.append(lr)
        t_bundle = get_random_timestep(min(model.scheduler.timesteps), max(model.scheduler.timesteps), (1, ))
        grad = torch.empty_like(latents)
        for _, t in enumerate(t_bundle):
            noise_pred, noise_randn = model.get_noise_pred(latents, text_embeddings, t, cfg)
            grad = grad + weights[t] * (noise_pred - noise_randn)
        latents = latents - 1e-2*grad
    images = model.decode_from_latents(latents)
    print(images.shape)
    #print(list_lr)
    return model.display_image(images)
    
if __name__ == "__main__":
    config = parse_yaml("lion.yaml")
    print(config)
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
    dict = {
            "batch_size": config['batch_size'],
            "prompt": [config['prompt']]*config['batch_size'],
            "negative_prompt": [config['negative_prompt']]*config['batch_size'],
            "pixel": config['pixel'],
            "num_steps": config['num_steps'],
            "cfg": config['guidance_scale']}
    print(dict['negative_prompt'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f"[INFO] Device {device} detected!")
    diffusion  = StableDiffusion(device, config)

    generation = diffusion.generate_img(dict)
    for i in range(len(generation)):
        img = generation[i]
        imageio.imwrite(f"Output/{i}.png", img)
    img = sds_sampling(diffusion, dict)
    for i in range(len(img)):
        imageio.imwrite(f"Output/test{i}.png", img[i])



