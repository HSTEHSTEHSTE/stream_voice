import yaml, os, shutil
from pathlib import Path
import torch
from dataset import Dataset

with open('config.yaml') as stream:
    configs = yaml.safe_load(stream)

# Create experiment folder
exp_path = os.path.join(configs['exp_path'], configs['exp_name'])
if os.path.isdir(exp_path):
    print("Warning: exp_path exists")
Path(exp_path).mkdir(parents=True, exist_ok=True)
## Copy configs into experiment folder
shutil.copyfile('config.yaml', os.path.join(exp_path, 'config.yaml'))
## Reload configs
with open(os.path.join(exp_path, 'config.yaml')) as stream:
    configs = yaml.safe_load(stream)
print(configs)

# verify device settings
if configs['device'] == 'cuda':
    assert "CUDA_VISIBLE_DEVICES" in os.environ and len(os.environ["CUDA_VISIBLE_DEVICES"]) > 0
    print('CUDA enabled')
print('CUDA devices: ', os.environ["CUDA_VISIBLE_DEVICES"])
device = torch.device(configs['device'])
