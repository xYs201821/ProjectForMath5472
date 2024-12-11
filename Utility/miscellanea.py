import torch
import yaml
import ast
import json
import os
import argparse

def save_metrics_json(metrics, work_dir, config, name='vsd'):
    """
    Save metrics as JSON with proper formatting
    """
    os.makedirs(work_dir, exist_ok=True)
    
    # Convert numpy arrays and floats to regular Python types
    json_metrics = {
        'training_info': {
            'seed': torch.initial_seed(),
            'size': config['size'],
            'num_steps': config['num_steps'],
            'lr': float(config['lr']),
            'phi_lr': float(config['phi_lr']),
            'cfg': float(config['cfg']),
            'prompt': config['prompt'],
            'negative_prompt': config['negative_prompt']
        },
        'metrics': {
            'losses': [float(x) for x in metrics['losses']],
            'epoch_times': [float(x) for x in metrics['epoch_times']],
            'unet_phi_times': [float(x) for x in metrics['unet_phi_times']],
            'particle_times': [float(x) for x in metrics['particle_times']],
            'peak_memory_usage': [float(x) for x in metrics['peak_memory_usage']],
            'learning_rates': [float(x) for x in metrics['learning_rates']]
        }
    }
    
    filename = os.path.join(work_dir, f"{name}_{torch.initial_seed()}_metrics.json")
    
    with open(filename, 'w') as f:
        json.dump(json_metrics, f, indent=4)
    
    print(f"[INFO] Metrics saved to: {filename}")
    return filename


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, required=False, default=0, help='Iteration number in a sequential script.')
    parser.add_argument('--config', type=str, required=False, default='config.yaml', help='Path to config file.')
    parser.add_argument('--prompt', type=str, required=False, default=None, help='Prompt.')
    parser.add_argument('--method', type=str, required=False, default='vsd', help=' all | pretrain | sds | vsd ')
    return parser.parse_args()

def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))

def parse_yaml(file_path):
    # Register the tuple constructor
    yaml.add_constructor('tag:yaml.org,2002:seq', tuple_constructor, Loader=yaml.SafeLoader)
    
    # Read and parse the YAML file
    with open(file_path, 'r') as file:
        parsed_data = yaml.safe_load(file)
    
    # Correct the parsing for lists and tuples
    corrected_data = {}
    for key, value in parsed_data.items():
        if isinstance(value, str):
            try:
                # Attempt to parse strings that look like tuples or lists
                parsed_value = ast.literal_eval(value)
                corrected_data[key] = parsed_value
            except (ValueError, SyntaxError):
                corrected_data[key] = value
        elif isinstance(value, list):
            corrected_data[key] = [v for v in value]
        else:
            corrected_data[key] = value
    
    return corrected_data

class DecayWeightScheduler:
    def __init__(self, 
                 initial_wd=1e-2,
                 final_wd=1e-3,
                 decay_type='expoential',
                 warmup_steps=1000,
                 total_steps=10000):
        self.initial_wd = initial_wd
        self.final_wd = final_wd
        self.decay_type = decay_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
    def get_decay_rate(self, step):
        if step < self.warmup_steps:
            return self.initial_wd * (step / self.warmup_steps)
            
        # Calculate progress after warmup
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        progress = torch.tensor(progress, dtype=torch.float16)
        if self.decay_type == 'linear':
            return self.initial_wd + (self.final_wd - self.initial_wd) * progress
            
        elif self.decay_type == 'exponential':
            return self.initial_wd * (self.final_wd / self.initial_wd) ** progress
            
        elif self.decay_type == 'cosine':
            return self.final_wd + 0.5 * (self.initial_wd - self.final_wd) * \
                   (1 + torch.cos(torch.pi * progress))
            
        elif self.decay_type == 'step':
            milestones = [0.3, 0.6, 0.9]
            decay_factor = 0.1
            for milestone in milestones:
                if progress >= milestone:
                    self.initial_wd *= decay_factor
            return self.initial_wd
            
        return self.initial_wd  