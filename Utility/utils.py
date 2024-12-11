import torch
import numpy as np

def get_random_timestep(low_t, high_t, size_t=(1,)):
    return torch.randint(max(low_t, 20), min(high_t, 980), size_t)

def get_noise_pred(unet, noise_latents, text_embeddings, t, cfg):
    noise_latents_model_input = torch.cat([noise_latents, noise_latents])
    noise_pred = unet(noise_latents_model_input, t, encoder_hidden_states=text_embeddings).sample
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)
    return noise_pred

def get_time_weights(model, type = "sqrt_cumprod"):
    if type == "sqrt_cumprod":
        weights = torch.sqrt(torch.cumprod(model.scheduler.alphas, dim=0)).to(model.device)    
    elif type == "cumprod":
        weights = torch.cumprod(model.scheduler.alphas, dim=0).to(model.device) 
    elif type == "minus":
        weights = (1 - model.scheduler.alphas).to(model.device)
    elif type == "minus_cumprod":
        weights = torch.cumprod(1 - model.scheduler.alphas, dim=0).to(model.device)
    elif type == "sqrt_minus_cumprod":
        weights = torch.sqrt(1 - torch.cumprod(input=model.scheduler.alphas, dim=0)).to(model.device)
    else:
        weights = torch.ones_like(model.scheduler.alphas).to(model.device)
    try:
        return weights.to(model.dtype)
    except:
        return weights