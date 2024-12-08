import torch
import yaml
import ast

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
                 initial_wd=0.05,
                 final_wd=0.001,
                 decay_type='linear',
                 warmup_steps=100,
                 total_steps=1000):
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
        progress = torch.tensor(progress)
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