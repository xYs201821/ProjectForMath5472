import imageio
import os

from Utility import parse_yaml, DecayWeightScheduler, UNet, LoRa
import torch
from torch import nn
from tqdm import tqdm
from Guidance import StableDiffusion
from diffusers import UNet2DConditionModel
torch.set_default_dtype(torch.float16)

def get_random_timestep(low_t, high_t, size_t=(1,)):
    return torch.randint(low_t, high_t, size_t)

def get_noise_pred(unet, noise_latents, text_embeddings, t, cfg):
    noise_latents_model_input = torch.cat([noise_latents, noise_latents])
    noise_pred = unet(noise_latents_model_input, t, encoder_hidden_states=text_embeddings).sample
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)
    return noise_pred

def get_time_weights(model, type = "sqrt_cumprod"):
    try:
        if type == "sqrt_cumprod":
            weights = torch.sqrt(torch.cumprod(model.scheduler.alphas, dim=0)).to(model.device)    
        elif type == "cumprod":
            weights = torch.cumprod(model.scheduler.alphas, dim=0).to(model.device) 
        elif type == "minus":
            weights = (1 - model.scheduler.alphas).to(model.device)
        elif type == "minus_cumprod":
            weights = torch.cumprod(1 - model.scheduler.alphas, dim=0).to(model.device)
        elif type == "sqrt_minus_cumprod":
            weights = torch.sqrt(1 - torch.cumprod(model.scheduler.alphas, dim=0)).to(model.device)
        else:
            weights = torch.tensor([1.0]*model.scheduler.timesteps).to(model.device)
    except:
        weights = torch.tensor([1.0]*model.scheduler.timesteps).to(model.device)
    return weights.to(model.dtype)

@torch.no_grad()
def sds_sampling(model, config):
    #preparation
    scheduler = DecayWeightScheduler()
    height, width = config["pixel"]
    cfg = config["cfg"]
    text_embeddings = model.get_text_embeddings(config["prompt"], config["negative_prompt"])
    train_steps = config["num_steps"]
    model.scheduler.set_timesteps(1000)
    
    #initiate random latents
    latents = torch.randn((config['batch_size'], model.unet.config.in_channels, height//8, width//8)).to(model.device)

    weights = get_time_weights(model, config["time_weight"])
    print(f"Start memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    #main training loop
    for itr in tqdm(range(train_steps)):
        if itr % 100 == 0:
            images = model.decode_from_latents(latents)
            imageio.imwrite(f"Test_sds/sds_{itr}.png",model.display_image(images)[0])
        lr = scheduler.get_decay_rate(itr)
        t_bundle = get_random_timestep(min(model.scheduler.timesteps), max(model.scheduler.timesteps), (1, ))  # single t, mini-batch
        grad = torch.empty_like(latents)
        for _, t in enumerate(t_bundle):
            noise_randn = torch.randn_like(latents).to(model.device)
            noise_latents = model.scheduler.add_noise(latents, noise_randn, t)
            noise_pred= get_noise_pred(model.unet, noise_latents, text_embeddings, t, cfg)
            grad = grad + weights[t] * (noise_pred - noise_randn)
        latents = latents - lr*grad
    print(f"Peak memory: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
        #list_grad_norm.append(torch.norm(grad)) norm of noise
    images = model.decode_from_latents(latents)
    return model.display_image(images)

def vsd_sampling(model, config):
    #preparation
    criterion = nn.MSELoss()
    scheduler = DecayWeightScheduler()
    height, width = config["pixel"]
    cfg = config["cfg"]
    unet_phi = config["unet_phi"]
    text_embeddings = model.get_text_embeddings(config["prompt"], config["negative_prompt"]).to(model.device)
    text_embeddings_phi = model.get_text_embeddings(config["prompt"], config["negative_prompt"]).to(model.device)
    train_steps = config["num_steps"]
    unet_phi.print_trainable_parameters()
    model.scheduler.set_timesteps(1000)
    metrics = {
        'losses': [],
        'grad_norms': [],
        'learning_rates': [],
        'peak_memory': []
    }
    #initiate random latents
    latents = torch.randn((config['batch_size'], model.unet.config.in_channels, height//8, width//8)).to(model.device)
    optimizer = torch.optim.AdamW(params=unet_phi.parameters(), lr=1e-4, weight_decay=0.01)
    
    weights = get_time_weights(model, config["time_weight"])
    print(f"Start memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    #main training loop
    for itr in tqdm(range(train_steps)):
        if itr % 100 == 0:
            images = model.decode_from_latents(latents)
            imageio.imwrite(f"Test_vsd/vsd_{itr}.png",model.display_image(images)[0])
            
        optimizer.zero_grad()
        #update particles
        lr = scheduler.get_decay_rate(itr)
        t = get_random_timestep(min(model.scheduler.timesteps), max(model.scheduler.timesteps)).to(model.device)
        noise_randn = torch.randn_like(latents, device=model.device)
        noise_latents = model.scheduler.add_noise(latents, noise_randn, t)
        with torch.no_grad():
            noise_pred= get_noise_pred(model.unet, noise_latents, text_embeddings, t, cfg)
        noise_nn_pred =  get_noise_pred(unet_phi, noise_latents.detach(), text_embeddings_phi.detach(), t, cfg)
        grad = weights[t] * (noise_pred - noise_nn_pred)
        latents = latents - lr*grad
        
        #use new particles to update unet_phi
        # t_phi = get_random_timestep(min(model.scheduler.timesteps), max(model.scheduler.timesteps)).to(model.device)
        # noise_randn_phi = torch.randn_like(latents, device=model.device)
        # noise_latents_phi = model.scheduler.add_noise(latents, noise_randn_phi, t_phi)
        # noise_pred_phi = get_noise_pred(unet_phi, noise_latents_phi, text_embeddings.detach(), t_phi, cfg)
        loss = criterion(noise_nn_pred, noise_randn) / config["batch_size"]
        loss.backward()
        optimizer.step()
        #metrics["losses"].append(loss.item())
    from peft import get_peft_model_state_dict

# Assuming 'model' is your LoRA-adapted model
    lora_state_dict = get_peft_model_state_dict(unet_phi)
    torch.save(lora_state_dict, f"lora_weights.pth")
    print(f"Peak memory: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
    images = model.decode_from_latents(latents)
    return model.display_image(images)

if __name__ == "__main__":
    config = parse_yaml("lion.yaml")
    #print(config)
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f"[INFO] Device {device} detected!")
    dict = {
            "batch_size": config['batch_size'],
            "prompt": [config['prompt']]*config['batch_size'],
            "negative_prompt": [config['negative_prompt']]*config['batch_size'],
            "pixel": config['pixel'],
            "num_steps": config['num_steps'],
            "cfg": config['guidance_scale'],
            "device": device,
            "unet_phi": UNet() if config["unet_phi"] == "unet" else LoRa(config["pretrain_model_path"],config["lora_rank"], config["lora_alpha"]).unet_phi,
            "time_weight": config["time_weight"]}
    diffusion = StableDiffusion(device, config)
    # generation = diffusion.generate_img(dict)
    # for i in range(len(generation)):
    #     img = generation[i]
    #     imageio.imwrite(f"Output/pre_{i}.png", img)
    # dict["cfg"] = 50
    # img = sds_sampling(diffusion, dict)
    # for i in range(len(img)):
    #     imageio.imwrite(f"Output/sds_{i}.png", img[i])
    img = vsd_sampling(diffusion, dict)
    for i in range(len(img)):
        imageio.imwrite(f"Output/vsd_{i}.png", img[i])
