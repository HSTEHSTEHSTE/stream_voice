import yaml, os, shutil
from pathlib import Path
import torch
from dataset import Dataset
from model.lm import StreamVoice
from tqdm import tqdm

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
print('Length of training dataset: ', len(dataset))

validation_dataset = Dataset(configs, set_name = 'dev')
validation_loader = torch.utils.data.DataLoader(dataset, 
    batch_size = configs['batch_size'], 
    shuffle = False,
    collate_fn = dataset.collate_fn, 
    drop_last = False, 
    num_workers = 16
)
print('Length of validation dataset: ', len(validation_dataset))

# Define model
model = torch.nn.DataParallel(StreamVoice(configs)).to(device)

# Define loss
cross_entropy_loss = torch.nn.CrossEntropyLoss()

# Define optimizer
optimizer = torch.optim.Adam(
    model.parameters(), 
    betas = configs['training']['optim']['betas'], 
    eps = configs['training']['optim']['eps'], 
    weight_decay = configs['training']['optim']['weight_decay']
)

for epoch_index in range(configs['training']['epoch']):
    print('Entering epoch ', epoch_index)
    for batch_index, batch in enumerate(loader):
        codec_pts = batch['codec_pts'].detach().to(device)
        asr_emb_pts = batch['asr_emb_pts'].detach().to(device)

        if not codec_pts.shape[1] % 4 == 0:
            codec_extension = torch.full(
                size = (codec_pts.shape[0], 4 - codec_pts.shape[1] % 4, codec_pts.shape[2]), 
                fill_value = configs['model']['codebook_num'] * configs['model']['codebook_dim']
            ).to(device)
            codec_pts = torch.cat([codec_pts, codec_extension.detach()], dim = 1).detach()

        output = model(codec_pts, asr_emb_pts) # [batch_size, seq_len, codebook_dim, codebook_num]
        output = output.view((output.shape[0], int(output.shape[1] / (configs['model']['frame_ratio'] + 1)), configs['model']['frame_ratio'] + 1, output.shape[2], output.shape[3]))
        output = output[:, :, 1:, :, :]
        output = torch.reshape(output, (output.shape[0], output.shape[1] * output.shape[2], output.shape[3], output.shape[4]))

        loss = None
        codec_pts = codec_pts[:, :int(configs['max_seq_len'] / (configs['model']['frame_ratio'] + 1) * configs['model']['frame_ratio']), :].detach()
        for codebook_num in range(configs['model']['codebook_num']):
            codec_pt = codec_pts[:, :, codebook_num] - configs['model']['codebook_dim'] * codebook_num
            mask = codec_pt != dataset.codec_size - configs['model']['codebook_dim'] * codebook_num
            codec_pt = torch.masked_select(codec_pt, mask)
            output_codec = torch.masked_select(output[:, :, :, codebook_num], mask.unsqueeze(2)).view((-1, configs['model']['codebook_dim']))
            if loss is None:
                loss = cross_entropy_loss(output_codec, codec_pt.detach())
            else:
                loss = loss + cross_entropy_loss(output_codec, codec_pt.detach())
        optimizer.zero_grad()
        loss.backward()

        # Clipping gradients to avoid gradient explosion
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), configs['training']['grad_clip_thresh'])

        optimizer.step()
        loss = loss.detach()

        if batch_index % configs['training']['report_every'] == 0:
            print('Batch: ', batch_index)
            print('Training loss: ', loss.item())

        if batch_index % configs['training']['valid_every'] == 0:
            model = model.eval()
            loss = 0
            for validation_batch_index, validation_batch in tqdm(enumerate(validation_loader), total = 100):
                if validation_batch_index >= 100:
                    break
                codec_pts = batch['codec_pts'].detach().to(device)
                asr_emb_pts = batch['asr_emb_pts'].detach().to(device)

                if not codec_pts.shape[1] % 4 == 0:
                    codec_extension = torch.full(
                        size = (codec_pts.shape[0], 4 - codec_pts.shape[1] % 4, codec_pts.shape[2]), 
                        fill_value = configs['model']['codebook_num'] * configs['model']['codebook_dim']
                    ).to(device)
                    codec_pts = torch.cat([codec_pts, codec_extension.detach()], dim = 1).detach()

                output = model(codec_pts, asr_emb_pts) # [batch_size, seq_len, codebook_dim, codebook_num]
                output = output.view((output.shape[0], int(output.shape[1] / (configs['model']['frame_ratio'] + 1)), configs['model']['frame_ratio'] + 1, output.shape[2], output.shape[3]))
                output = output[:, :, 1:, :, :]
                output = torch.reshape(output, (output.shape[0], output.shape[1] * output.shape[2], output.shape[3], output.shape[4]))

                codec_pts = codec_pts[:, :int(configs['max_seq_len'] / (configs['model']['frame_ratio'] + 1) * configs['model']['frame_ratio']), :].detach()
                for codebook_num in range(configs['model']['codebook_num']):
                    codec_pt = codec_pts[:, :, codebook_num] - configs['model']['codebook_dim'] * codebook_num
                    mask = codec_pt != dataset.codec_size - configs['model']['codebook_dim'] * codebook_num
                    codec_pt = torch.masked_select(codec_pt, mask)
                    output_codec = torch.masked_select(output[:, :, :, codebook_num], mask.unsqueeze(2)).view((-1, configs['model']['codebook_dim']))
                    loss += cross_entropy_loss(output_codec, codec_pt.detach()).item()
            print('Validation loss: ', loss / 100)


            model = model.train()