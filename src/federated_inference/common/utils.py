import torch

def cuba_info():
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

import torch
import torch.nn as nn
import json

def summarize_model_to_json(model, input_size, device='cpu', json_path='model_summary.json'):
    model = model.to(device)
    model.eval()

    summary = []
    
    def register_hook(module):
        def hook(module, input, output):
            summary.append({
                'layer': module.__class__.__name__,
                'input_shape': list(input[0].size()) if input else [],
                'output_shape': list(output.size()) if isinstance(output, torch.Tensor) else [str(type(output))],
                'num_parameters': sum(p.numel() for p in module.parameters() if p.requires_grad)
            })
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    hooks = []
    model.apply(register_hook)

    dummy_input = torch.randn(1, *input_size).to(device)
    model(dummy_input)  # Run forward pass to trigger hooks

    for h in hooks:
        h.remove()

    # Save summary as JSON
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=4)

    return summary


import torch 
import random 
import numpy as np
    
def set_seed(seed=42):
    torch.manual_seed(seed)                # CPU
    torch.cuda.manual_seed(seed)           # Current GPU
    torch.cuda.manual_seed_all(seed)       # All GPUs
    np.random.seed(seed)                   # NumPy
    random.seed(seed)                      # Python random
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

