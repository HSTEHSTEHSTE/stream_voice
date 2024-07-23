import yaml, os, shutil
from pathlib import Path
import torch
from dataset import Dataset
from model.lm import StreamVoice
from utils import get_checkpoints, update_checkpoints
from optimizer import ScheduledOptim
from tqdm import tqdm

with open('config.yaml') as stream:
    configs = yaml.safe_load(stream)

# Create experiment folder
exp_path = os.path.join(configs['exp_path'], configs['exp_name'])
checkpoint_path = os.path.join(exp_path, 'ckpt')
if os.path.isdir(exp_path):
    print("Warning: exp_path exists", flush = True)
Path(checkpoint_path).mkdir(parents = True, exist_ok = True)
## Copy configs into experiment folder
shutil.copyfile('config.yaml', os.path.join(exp_path, 'config.yaml'))
## Reload configs
with open(os.path.join(exp_path, 'config.yaml')) as stream:
    configs = yaml.safe_load(stream)
print(configs, flush = True)

# verify device settings
if configs['device'] == 'cuda':
    assert "CUDA_VISIBLE_DEVICES" in os.environ and len(os.environ["CUDA_VISIBLE_DEVICES"]) > 0
    print('CUDA enabled', flush = True)
print('CUDA devices: ', os.environ["CUDA_VISIBLE_DEVICES"], flush = True)
device = torch.device(configs['device'])

# Get dataset
dataset = Dataset(configs, set_name = 'train')
loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size = configs['batch_size'], 
    shuffle = (configs['training']['train_length_limit'] >= len(dataset)),
    collate_fn = dataset.collate_fn, 
    drop_last = True, 
    num_workers = 16
)
print('Length of training dataset: ', len(dataset), flush = True)

validation_dataset = Dataset(configs, set_name = 'dev')
validation_loader = torch.utils.data.DataLoader(dataset, 
    batch_size = configs['batch_size'], 
    shuffle = False,
    collate_fn = dataset.collate_fn, 
    drop_last = False, 
    num_workers = 16
)
print('Length of validation dataset: ', len(validation_dataset), flush = True)

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

# Load checkpoint if possible
current_step = 0
checkpoints = get_checkpoints(checkpoint_path)
restore_checkpoint = -1
if configs['training']['restore_step'] in checkpoints.keys() or configs['training']['restore_step'] == 'infer':
    if configs['training']['restore_step'] in checkpoints.keys():
        restore_checkpoint = configs['training']['restore_step']
    if configs['training']['restore_step'] == 'infer':
        restore_checkpoint = max(checkpoints.keys())
    checkpoint = torch.load(os.path.join(
        checkpoint_path, 'checkpoint_{}.pt'.format(restore_checkpoint)))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("\n---Model Restored at Step {}---\n".format(restore_checkpoint))
    current_step = restore_checkpoint

# Define optimizer scheduler
scheduled_optimizer = ScheduledOptim(
    optimizer = optimizer, 
    d_model = configs['model']['transformer_dim'], 
    n_warmup_steps = configs['training']['n_warmup_step'], 
    current_steps = (restore_checkpoint + 1) / configs['training']['gradient_acc_steps'],
    init_lr = configs['training']['optim']['init_lr']
)

for epoch_index in range(configs['training']['epoch']):
    print('Entering epoch ', epoch_index, flush = True)
    for batch_index, batch in enumerate(loader):
        if batch_index >= configs['training']['train_length_limit']:
            break
        codec_pts = batch['codec_pts'].detach().to(device)
        asr_emb_pts = batch['asr_emb_pts'].detach().to(device)

        if not codec_pts.shape[1] % configs['model']['frame_ratio'] == 0:
            codec_extension = torch.full(
                size = (codec_pts.shape[0], configs['model']['frame_ratio'] - codec_pts.shape[1] % configs['model']['frame_ratio'], codec_pts.shape[2]), 
                fill_value = configs['model']['codebook_num'] * configs['model']['codebook_dim']
            ).to(device)
            codec_pts = torch.cat([codec_pts, codec_extension.detach()], dim = 1).detach()

        codec_extra_pad = torch.full(
            size = (codec_pts.shape[0], 1, codec_pts.shape[2]), 
            fill_value = configs['model']['codebook_num'] * configs['model']['codebook_dim']
        ).to(device)
        in_codec_pts = torch.cat([codec_pts, codec_extra_pad.detach()], dim = 1).detach()

        output = model(in_codec_pts, asr_emb_pts) # [batch_size, seq_len, codebook_dim, codebook_num]
        output = output[:, :-1, :, :]
        output = output.view((output.shape[0], int(output.shape[1] / (configs['model']['frame_ratio'] + 1)), configs['model']['frame_ratio'] + 1, output.shape[2], output.shape[3]))
        output = output[:, :, :-1, :, :]
        output = torch.reshape(output, (output.shape[0], output.shape[1] * output.shape[2], output.shape[3], output.shape[4]))

        codec_pts = codec_pts[:, :int(configs['max_seq_len'] / (configs['model']['frame_ratio'] + 1) * configs['model']['frame_ratio']), :].detach()
        loss = torch.tensor(0.).to(device)
        for codebook_num in range(configs['model']['codebook_num']):
            codec_pt = codec_pts[:, :, codebook_num] - configs['model']['codebook_dim'] * codebook_num
            mask = codec_pt != dataset.codec_size - configs['model']['codebook_dim'] * codebook_num
            codec_pt = torch.masked_select(codec_pt, mask)
            output_codec = torch.masked_select(output[:, :, :, codebook_num], mask.unsqueeze(2)).view((-1, configs['model']['codebook_dim']))
            loss = loss + cross_entropy_loss(output_codec, codec_pt.detach())
        
        loss = loss / configs['training']['gradient_acc_steps']
        loss.backward()
        
        if (current_step + 1) % configs['training']['gradient_acc_steps'] == 0:
            # Clipping gradients to avoid gradient explosion
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), configs['training']['grad_clip_thresh'])

            scheduled_optimizer.step_and_update_lr()
            scheduled_optimizer.zero_grad()
            loss = loss.detach()

        if (current_step + 1) % configs['training']['report_every'] == 0:
            print('Batch: ', batch_index, flush = True)
            print('Training loss: ', loss.item() * configs['training']['gradient_acc_steps'], flush = True)
            print('Learning rate: ', scheduled_optimizer.lr, flush = True)

        if (current_step + 1) % configs['training']['valid_every'] == 0:
            model = model.eval()
            loss = 0
            for validation_batch_index, validation_batch in tqdm(enumerate(validation_loader), total = min(configs['training']['valid_length_limit'], len(validation_loader))):
                if validation_batch_index >= configs['training']['valid_length_limit']:
                    break
                codec_pts = batch['codec_pts'].detach().to(device)
                asr_emb_pts = batch['asr_emb_pts'].detach().to(device)

                if not codec_pts.shape[1] % configs['model']['frame_ratio'] == 0:
                    codec_extension = torch.full(
                        size = (codec_pts.shape[0], configs['model']['frame_ratio'] - codec_pts.shape[1] % configs['model']['frame_ratio'], codec_pts.shape[2]), 
                        fill_value = configs['model']['codebook_num'] * configs['model']['codebook_dim']
                    ).to(device)
                    codec_pts = torch.cat([codec_pts, codec_extension.detach()], dim = 1).detach()

                codec_extra_pad = torch.full(
                    size = (codec_pts.shape[0], 1, codec_pts.shape[2]), 
                    fill_value = configs['model']['codebook_num'] * configs['model']['codebook_dim']
                ).to(device)
                in_codec_pts = torch.cat([codec_pts, codec_extra_pad.detach()], dim = 1).detach()

                output = model(in_codec_pts, asr_emb_pts) # [batch_size, seq_len, codebook_dim, codebook_num]
                output = output[:, :-1, :, :]
                output = output.view((output.shape[0], int(output.shape[1] / (configs['model']['frame_ratio'] + 1)), configs['model']['frame_ratio'] + 1, output.shape[2], output.shape[3]))
                output = output[:, :, :-1, :, :]
                output = torch.reshape(output, (output.shape[0], output.shape[1] * output.shape[2], output.shape[3], output.shape[4]))

                codec_pts = codec_pts[:, :int(configs['max_seq_len'] / (configs['model']['frame_ratio'] + 1) * configs['model']['frame_ratio']), :].detach()
                for codebook_num in range(configs['model']['codebook_num']):
                    codec_pt = codec_pts[:, :, codebook_num] - configs['model']['codebook_dim'] * codebook_num
                    mask = codec_pt != dataset.codec_size - configs['model']['codebook_dim'] * codebook_num
                    codec_pt = torch.masked_select(codec_pt, mask)
                    output_codec = torch.masked_select(output[:, :, :, codebook_num], mask.unsqueeze(2)).view((-1, configs['model']['codebook_dim']))
                    loss += cross_entropy_loss(output_codec, codec_pt.detach()).item()
                del output
                del output_codec
            print('Validation loss: ', loss / min(configs['training']['valid_length_limit'], len(validation_loader)), flush = True)

            model = model.train()

        if (current_step + 1) % configs['training']['save_every'] == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, 
                os.path.join(checkpoint_path, 'checkpoint_{}.pt'.format(current_step)))
            checkpoints[current_step] = 'checkpoint_{}.pt'.format(current_step)
            update_checkpoints(checkpoint_path, checkpoints, configs['training']['keep_checkpoints'])
            print("save model at step {} ...".format(current_step), flush = True)
            breakpoint()

        current_step += 1