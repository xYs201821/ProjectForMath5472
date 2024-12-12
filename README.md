# Project For MATH 5472

This repository maintains an elementary implementation of my course project for MATH5472: [2D experiments for VSD and SDS methods](report.pdf).

## Requirements

```bash
pip install -r requirements.txt
```

## Project Structure

```bash
├── config.yaml
├── main.py
├── sds.py
├── vsd.py
├── Guidance/
│   └── sd_pipeline.py
└── Utility/
    ├── lora.py
    ├── miscellanea.py
    └── utils.py 
```

## Configuration

Create a YAML config file (e.g. config.yaml )with the following parameters:

```yaml
# general
prompt: "Pancakes on a plate." # prompt that provides in commend line will overwrite prompt in YAML file
negative_prompt: ""
size: 1             # number of particles
pixel: (512, 512)   # width, height
num_steps: 2500     # Number of epochs
guidance_scale: 7.5 # Classifier Free Guidance sacle
lr: 3e-2            # Learning rate of particles
seed: 42            
time_weight: "sqrt_minus_cumprod" # | minus_cumprod | minus | cumprod | sqrt_minus_cumprod | sqrt_cumprod | 
work_dir: "Output/" # Output directory

#lora params
unet_phi: lora
lora_rank: 32
lora_alpha: 256
phi_lr: 1e-4

#pretrain_model
pretrain_model_id:  "CompVis/stable-diffusion-v1-4"
# pretrain_model_path: "***/CompVis/stable-diffusion-v1-4/"
# local path is of high priority
```

## Usage

Examples are also provided in [JupyterNotebook](sample.ipynb)

### Basic Usage

```bash
python main.py --config config.yaml
```

### Custom Prompt

```bash
python main.py --config config.yaml --prompt "your custom prompt"
```

## Output Structure

The script generates images in three directories under the specified `work_dir`:

- `pre/`: Images generated using pretrained model (Stable-Diffusion).
- `sds/`: Images generated using SDS sampling
- `vsd/`: Images generated using VSD sampling

Images are saved with numerical indices (e.g., `0000.png`, `0001.png`).

## Command Line Arguments

- `--config`: Path to the configuration YAML file. (required, default='config.yaml')
- `--method`: Selected sampling method, one of ['all', 'pretrain', 'sds', 'vsd']. (optional, default='vsd')
- `--prompt`: Override the prompt in YAML file. (optional, default=None)
- `--iter`:   Number of current process for multiple runs. (not required)

## Notes

1. For generating 4 particles of pixel sizes = (512, 512), the peak memory usage is about 6GB. The average training time is around 15 mins. (not yet optimized, quite inefficient)
2. The script automatically uses CUDA if available, otherwise falls back to CPU.
3. Images are generated in latent space and then decoded to pixel space.
4. The seed is automatically adjusted for different processes identified by "--iter" to ensure variety.

## Reference Implementation

[1]: [Zhengyi Wang, Cheng Lu, Yikai Wang, et al. "Prolific Dreamer"](https://github.com/thu-ml/prolificdreamer )

[2]: [Yuanzhi Zhu, "Unofficial implementation of 2D prolific_dreamer"](https://github.com/yuanzhi-zhu/prolific_dreamer2d )

[3]: [Justin Wu, "DreamFusioncc"](https://github.com/chinhsuanwu/dreamfusionacc/tree/master)
