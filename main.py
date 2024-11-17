import imageio
import os

from Utility.miscellanea import read_config
# suppress partial model loading warning
import torch
from Guidance import StableDiffusion

if __name__ == "__main__":
    dict = {"prompt": ["a beautiful and powerful mysterious sorceress, smile, sitting on a rock, lightning magic, hat, detailed leather clothing with gemstones, dress, castle background, digital art, hyperrealistic, fantasy, dark art, artstation"],
            "negative_prompt": [""],
            "pixel": (512, 512),
            "num_steps": 100,
            "cfg": 7.5}
    config = read_config("lion.yaml")
    if config.seed is not None:
        torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device {device} detected!")
    diffusion1  = StableDiffusion(device, config)
    generation = diffusion1.generate_img(dict)
    print(generation)
    for i in range(len(generation)):
        img = generation[i]
        imageio.imwrite(f"Output/{i}.png", img)


