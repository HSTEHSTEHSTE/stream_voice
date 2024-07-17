import yaml, os, shutil
from pathlib import Path
import torch
from dataset import Dataset
from model.lm import StreamVoice

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

# Get dataset
dataset = Dataset(configs, set_name = 'train')
loader = torch.utils.data.DataLoader(dataset, batch_size = configs['batch_size'], shuffle = True,
                    collate_fn = dataset.collate_fn, drop_last = True, num_workers = 16)

# Define model
model = torch.nn.DataParallel(StreamVoice(configs)).to(device)

for i, batch in enumerate(loader):
    codec_pts = batch['codec_pts'].to(device)
    codec_lens = batch['codec_lens'].to(device)
    asr_emb_pts = batch['asr_emb_pts'].to(device)
    asr_emb_lens = batch['asr_emb_lens'].to(device)

    if not codec_pts.shape[1] % 4 == 0:
        codec_extension = torch.full(
            size = (codec_pts.shape[0], 4 - codec_pts.shape[1] % 4, codec_pts.shape[2]), 
            fill_value = configs['model']['codebook_num'] * configs['model']['codebook_dim']
        ).to(device)
        codec_pts = torch.cat([codec_pts, codec_extension], dim = 1)

    model(codec_pts, asr_emb_pts)
