import logging
import time
import os
import torch
import torch.nn as nn
import imageio
import numpy as np
from tqdm import tqdm
from Utility import DecayWeightScheduler, save_metrics_json, get_time_weights, get_random_timestep, get_noise_pred

def vsd_sampling(model, init_particles, config):
    #logging
    try:
        logging.basicConfig(
        filename=os.path.join(f'vsd.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    except:
        pass
    metrics = {
        'start_time': time.time(),
        'training_time': .0,
        'losses': [],
        'epoch_times': [],
        'unet_phi_times': [],
        'particle_times': [],
        'peak_memory_usage': [],
        'learning_rates': []
    }
    
    #preparation
    criterion = nn.MSELoss()
    scheduler = DecayWeightScheduler(config["lr"])
    cfg = config["cfg"]
    unet_phi = config["unet_phi"]
    text_embeddings_phi = model.get_text_embeddings(config["prompt"], config["negative_prompt"])
    train_steps = config["num_steps"]
    model.scheduler.set_timesteps(1000)

    #initiate particles and unet_phi optimizer
    particles = init_particles.clone()
    optimizer = torch.optim.AdamW(params=unet_phi.parameters(), lr=config["phi_lr"], weight_decay=0.01)
    weights = get_time_weights(model, config["time_weight"])
    print(f"Start memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    #main training loop
    for itr in tqdm(range(train_steps*config["size"]), disable=config["validation"]):
        lr = scheduler.get_decay_rate(itr)
        t = get_random_timestep(min(model.scheduler.timesteps), max(model.scheduler.timesteps), size_t=(1, )).to(model.device)
        unet_phi.zero_grad()
        optimizer.zero_grad()
        indices = torch.randint(0, particles.shape[0], (1, ))
        latents = particles[indices]
        
        if itr % (25*config["size"]) == 0:     
            losses = [] 
            lrs = [] 
            step_particle_times = []
            step_unet_phi_times = []
            step_times = []
            torch.cuda.reset_peak_memory_stats()
        step_start = time.time()  
        
        #update particles
        with torch.no_grad():
            noise_randn = torch.randn_like(latents, device=model.device)
            noise_latents = model.scheduler.add_noise(latents, noise_randn, t)
            noise_pred = get_noise_pred(model.unet, noise_latents, text_embeddings_phi, t, cfg)
        noise_nn_pred =  get_noise_pred(unet_phi, noise_latents, text_embeddings_phi, t, cfg)
        grad = weights[t] * (noise_pred - noise_nn_pred)
        particles[indices] = particles[indices] - lr*grad
        
        step_middle = time.time()
        step_particle_times.append(time.time() - step_start)
        
        #update unet_phi
        loss = criterion(noise_nn_pred, noise_randn)
        loss.backward()
        optimizer.step()
        
        #logging
        losses.append(loss.item())
        lrs.append(lr)
        step_unet_phi_times.append(time.time() - step_middle)
        step_times.append(time.time() - step_start)
        with torch.no_grad():
            if itr % (250*config["size"]) == 0:
                if config["test"]:
                    images = model.decode_from_latents(particles[0, :, :, :].unsqueeze(0))
                    imageio.imwrite(os.path.join(f"Test{cfg}/vsd", f"{itr//config["size"]:05d}.png"),model.display_image(images)[0])
            if  itr % (25*config["size"]) == 0:
                metrics['losses'].append(np.mean(losses))
                metrics['learning_rates'].append(np.mean(lrs))
                metrics['epoch_times'].append(np.mean(step_times))
                metrics['peak_memory_usage'].append(torch.cuda.max_memory_allocated()/1024**3)
                metrics['unet_phi_times'].append(np.mean(step_unet_phi_times)) 
                metrics['particle_times'].append(np.mean(step_particle_times)) 
                try:
                    logging.info(f"Step {itr//config['size']}-{itr//config['size'] + 25}: Avg Loss={np.mean(losses):.6f}, "
                    f"Avg Step Time={np.mean(step_times):.3f}s, "
                    f"Avg Learning Rate={np.mean(lrs):.6f}, "
                    f"Peak Memory Usage={torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
                except:
                    pass
        try:
            del noise_latents, noise_pred, noise_nn_pred, grad, loss
        except:
            pass
    metrics['training_time'] = metrics['training_time'] + time.time() - metrics['start_time']
    try:
        save_metrics_json(metrics, config["work_dir"], config, name='vsd')
    except:
        pass
    return particles