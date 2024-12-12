import os
import torch
import imageio
from tqdm import tqdm
from Utility import DecayWeightScheduler, get_time_weights, get_random_timestep, get_noise_pred

@torch.no_grad()
def sds_sampling(model, init_particles, config):
    #preparation
    scheduler = DecayWeightScheduler(initial_wd=config["lr"])
    try:
        cfg = config["cfg"] if not config["validation"] else 50
    except:
        cfg = config["cfg"]
    text_embeddings = model.get_text_embeddings(config["prompt"]*config["size"], config["negative_prompt"]*config["size"])
    train_steps = config["num_steps"]
    model.scheduler.set_timesteps(1000)
    
    #initiate random latents
    particles = init_particles.clone()
    weights = get_time_weights(model, config["time_weight"])
    print(f"Start memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    #main training loop
    for itr in tqdm(range(train_steps), disable=config["validation"]):
        torch.cuda.empty_cache()
        if itr % 250 == 0:
            if config['test']:
                images = model.decode_from_latents(particles[0, :, :, :].unsqueeze(0))
                imageio.imwrite(os.path.join(f"Test{cfg}/sds", f"{itr:05d}.png"),model.display_image(images)[0])
        lr = scheduler.get_decay_rate(itr)
        latents = particles
        #update particles
        t = get_random_timestep(min(model.scheduler.timesteps), max(model.scheduler.timesteps), size_t=(1, )).to(model.device)
        noise_randn = torch.randn_like(input=latents).to(model.device)
        noise_latents = model.scheduler.add_noise(latents, noise_randn, t)
        noise_pred= get_noise_pred(model.unet, noise_latents, text_embeddings, t, cfg)
        grad = weights[t] * (noise_pred - noise_randn)
        particles = particles - lr*grad
        try:
            del noise_latents, noise_pred, grad, noise_randn, loss
        except:
            pass
    torch.cuda.empty_cache()
    return particles