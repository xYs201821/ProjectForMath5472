# Stable Diffusion Sampling with VSD/SDS

This repository implements Vector Stochastic Dynamics (VSD) and Score Distillation Sampling (SDS) for Stable Diffusion image generation.

## Requirements

```bash
pip install torch imageio pyyaml
```

## Project Structure

```
.
├── config/
│   └── default.yaml
├── main.py
├── sds.py
├── vsd.py
├── Guidance.py
└── Utility.py
```

## Configuration

Create a YAML config file with the following parameters:

```yaml
# Model Configuration
pretrain_model_path: "path/to/stable-diffusion-model"
lora_rank: 4
lora_alpha: 4

# Generation Parameters
seed: 42
size: 4  # number of images to generate
pixel: [512, 512]  # image dimensions
num_steps: 50
guidance_scale: 7.5
lr: 0.01
phi_lr: 0.01

# Prompt Configuration
prompt: "your prompt here"
negative_prompt: "your negative prompt here"

# Output Configuration
work_dir: "outputs/"

# Optional Parameters
time_weight: 1.0
test: false
validation: false
```

## Usage

### Basic Usage

```bash
python main.py --config config/default.yaml
```

### Custom Prompt

```bash
python main.py --config config/default.yaml --prompt "your custom prompt"
```

### Multiple Iterations

```bash
python main.py --config config/default.yaml --iter 0
```

## Output Structure

The script generates images in three directories under the specified `work_dir`:

- `pre/`: Initial generated images
- `sds/`: Images generated using SDS sampling
- `vsd/`: Images generated using VSD sampling

Images are saved with numerical indices (e.g., `0000.png`, `0001.png`).

## Command Line Arguments

- `--config`: Path to the configuration YAML file (required)
- `--prompt`: Override the prompt in config file (optional)
- `--iter`: Iteration number for multiple runs (optional, default=0)

## Features

- Supports both CPU and CUDA devices
- Implements LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Configurable sampling parameters
- Multiple sampling methods (VSD/SDS)
- Batch image generation
- Seed control for reproducibility

## Notes

1. Make sure you have sufficient GPU memory for the specified image size and batch size
2. The script automatically uses CUDA if available, otherwise falls back to CPU
3. Images are generated in latent space and then decoded to pixel space
4. The seed is automatically adjusted for different iterations to ensure variety

## Customization

To modify the sampling behavior, you can:
1. Uncomment the SDS sampling section in `main.py`
2. Adjust the learning rates (`lr` and `phi_lr`) in the config
3. Modify the number of steps (`num_steps`) for different quality-speed tradeoffs
4. Change the guidance scale for stronger/weaker prompt adherence