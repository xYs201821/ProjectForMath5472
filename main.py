import imageio
import os
from sds import sds_sampling
from vsd import vsd_sampling
from Utility import parse_yaml, LoRa, parse_args
import torch
from Guidance import StableDiffusion
torch.set_default_dtype(torch.float16)
def main():
    args = parse_args()

    try:
        config = parse_yaml(args.config)
        if args.prompt is not None:
            config["prompt"] = args.prompt
        print(f"[INFO] Load {args.config}: \n {config}!")
    except:
        print(f"Config file {args.config} not found!")
        exit()
    try:
        # Use the iteration number
        print(f"Current iteration: {args.iter}")
        if config['seed'] is not None:
            torch.manual_seed(config['seed'] + args.iter*201821)
    except:
        print(f"Current iteration: 0")
        if config['seed'] is not None:
            torch.manual_seed(config['seed'])
    print(f"[INFO] Seed {torch.initial_seed()} detected!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f"[INFO] Device {device} detected!")
    dict = {
            "seed": config["seed"],
            "size": config["size"],
            "prompt": [args.prompt if args.prompt is not None else config["prompt"]],
            "negative_prompt": [config["negative_prompt"]],
            "pixel": config["pixel"],
            "num_steps": config["num_steps"],
            "cfg": config["guidance_scale"],
            "device": device,
            "lr": config["lr"],
            "phi_lr": config["phi_lr"],
            "unet_phi": LoRa(config["pretrain_model_path"], config["lora_rank"], config["lora_alpha"]).unet_phi,
            "time_weight": config["time_weight"],
            "work_dir": config["work_dir"]}
    try:
        if config["test"] is not None: 
            dict["test"] = config["test"]
    except:
        dict["test"] = False
    try:
        if config["validation"] is not None:
            dict["validation"] = config["validation"]
    except:
        dict["validation"] = False
    model = StableDiffusion(device, config=config)
    init_particles = torch.randn((config['size'], model.unet.config.in_channels, config["pixel"][0]//8, config["pixel"][1]//8)).to(model.device)
    
    output_dir = config["work_dir"]
    os.makedirs(output_dir, exist_ok=True)

    def save_img(latents, output_dir):
        for i in range(latents.shape[0]):
            img = model.decode_from_latents(latents[i].unsqueeze(0))
            img = model.display_image(img)
            imageio.imwrite(os.path.join(output_dir, f"{(i+args.iter*config["size"]):04d}.png"), img[0])
    
    if args.method in ['pre', 'all']: 
        output_pre = os.path.join(output_dir, "pre")
        os.makedirs(output_pre, exist_ok=True)  
        img_latents = model.generate_img(init_particles, dict)
        save_img(img_latents, output_dir)
    if args.method in ['sds', 'all']: 
        output_sds = os.path.join(output_dir, "sds")
        os.makedirs(output_sds, exist_ok=True)
        img_latents = sds_sampling(model, init_particles, dict)
        save_img(img_latents, output_sds)
    if args.method in ['vsd', 'all']: 
        output_vsd = os.path.join(output_dir, "vsd")
        os.makedirs(output_vsd, exist_ok=True)
        img_latents = vsd_sampling(model, init_particles, dict)
        save_img(img_latents, output_vsd)
    

if __name__ == "__main__":
    main()