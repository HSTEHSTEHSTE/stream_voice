import yaml, os, shutil
from pathlib import Path
import torch, torchaudio
from dataset import Dataset
from model.lm import StreamVoice
from tqdm import tqdm
from model.audiodec import AudioDec, assign_model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', '-e', type = str, required = True)
parser.add_argument('--restore-step', '-r', type = int, default = 0)
parser.add_argument('--prompt_len', '-p', type = int, default = 100)

args = parser.parse_args()

with open('config.yaml') as stream:
    configs = yaml.safe_load(stream)
exp_path = exp_path = os.path.join(configs['exp_path'], args.exp_name)
checkpoint_path = os.path.join(exp_path, 'ckpt')
# Load configs
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
dataset = Dataset(configs, set_name = 'test')
loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size = 1, 
    shuffle = False,
    collate_fn = dataset.collate_fn, 
    drop_last = False, 
    num_workers = 16
)
print('Length of test dataset: ', len(dataset))

# Define model
model = torch.nn.DataParallel(StreamVoice(configs)).to(device)
sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model('libritts_v1')
audiodec = AudioDec(tx_device = device, rx_device = device)
audiodec.load_transmitter(encoder_checkpoint)
audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)

# Define loss
cross_entropy_loss = torch.nn.CrossEntropyLoss()

# Load model
checkpoint = torch.load(os.path.join(
    checkpoint_path, 'checkpoint_{}.pt'.format(args.restore_step)))
model.load_state_dict(checkpoint['model'])
del checkpoint
print("\n---Model Restored at Step {}---\n".format(args.restore_step))
model = model.eval()

def advance_one_step(codec_prompt, codec_extension, codec_extra_pad, model, asr_emb_prompt, current_codec_pos, outputs):
    
    codec_prompt_ext = torch.cat([codec_prompt, codec_extension.detach()], dim = 1).detach()
    codec_prompt_ext = torch.cat([codec_prompt_ext, codec_extra_pad.detach()], dim = 1).detach()

    output = model(codec_prompt_ext.detach(), asr_emb_prompt.detach())
    del codec_prompt_ext
    output = output[:, :-1, :, :]
    output = output.view((output.shape[0], int(output.shape[1] / (configs['model']['frame_ratio'] + 1)), configs['model']['frame_ratio'] + 1, output.shape[2], output.shape[3]))
    output = output[:, :, :-1, :, :]
    output = torch.reshape(output, (output.shape[0], output.shape[1] * output.shape[2], output.shape[3], output.shape[4]))
    outputs.append(output[:, current_codec_pos, :, :].detach())
    next_tokens = torch.argmax(output[:, current_codec_pos, :, :], dim = 1).unsqueeze(1)
    for codebook_num in range(configs['model']['codebook_num']):
        next_tokens[:, :, codebook_num] += codebook_num * configs['model']['codebook_dim']
    codec_prompt = torch.cat([codec_prompt, next_tokens], dim = 1)
    del output
    return codec_prompt, outputs, next_tokens

for batch_index, batch in enumerate(loader):
    codec_pts = batch['codec_pts'].detach().to(device)
    asr_emb_pts = batch['asr_emb_pts'].detach().to(device)

    codec_prompt_len = int(args.prompt_len / (configs['model']['frame_ratio'] + 1) * configs['model']['frame_ratio'])
    if codec_pts.shape[1] > codec_prompt_len:
        outputs = []
        codec_prompt = codec_pts[:, :codec_prompt_len, :]
        codec_prompt_in = codec_pts[:, :codec_prompt_len, :]
        asr_emb_prompt_len = int(args.prompt_len / (configs['model']['frame_ratio'] + 1))
        asr_emb_prompt = asr_emb_pts[:, :asr_emb_prompt_len, :]
        current_codec_pos = codec_prompt_len
        current_asr_emb_pos = asr_emb_prompt_len
        codec_extra_pad = torch.full(
            size = (codec_pts.shape[0], 1, codec_pts.shape[2]), 
            fill_value = configs['model']['codebook_num'] * configs['model']['codebook_dim']
        ).to(device)
        while current_codec_pos < codec_pts.shape[1] and current_codec_pos < int((configs['max_seq_len'] - 1) / (configs['model']['frame_ratio'] + 1) * configs['model']['frame_ratio']):
            if not codec_prompt.shape[1] % configs['model']['frame_ratio'] == 0:
                codec_extension = torch.full(
                    size = (codec_prompt.shape[0], configs['model']['frame_ratio'] - codec_prompt.shape[1] % configs['model']['frame_ratio'], codec_prompt.shape[2]), 
                    fill_value = configs['model']['codebook_num'] * configs['model']['codebook_dim']
                ).to(device)
                _, outputs, next_tokens = advance_one_step(codec_prompt_in, codec_extension, codec_extra_pad, model, asr_emb_prompt, current_codec_pos, outputs)
                codec_prompt = torch.cat([codec_prompt, next_tokens], dim = 1)
                current_codec_pos += 1
                codec_prompt_in = codec_pts[:, :current_codec_pos, :]
                # codec_prompt_in = codec_prompt
                del codec_extension
            else:
                if int(codec_prompt.shape[1] / configs['model']['frame_ratio']) == current_asr_emb_pos:
                    # next token is asr_emb, skip forward
                    current_asr_emb_pos += 1
                    asr_emb_prompt = asr_emb_pts[:, :current_asr_emb_pos, :]
                else:
                    codec_extension = torch.full(
                        size = (codec_prompt.shape[0], configs['model']['frame_ratio'], codec_prompt.shape[2]), 
                        fill_value = configs['model']['codebook_num'] * configs['model']['codebook_dim']
                    ).to(device)
                    _, outputs, next_tokens = advance_one_step(codec_prompt_in, codec_extension, codec_extra_pad, model, asr_emb_prompt, current_codec_pos, outputs)
                    codec_prompt = torch.cat([codec_prompt, next_tokens], dim = 1)
                    current_codec_pos += 1
                    codec_prompt_in = codec_pts[:, :current_codec_pos, :]
                    # codec_prompt_in = codec_prompt
                    del codec_extension
        del codec_extra_pad
    del asr_emb_pts
    loss = 0
    outputs = torch.stack(outputs, dim = 1)
    for codebook_num in range(configs['model']['codebook_num']):
        codec_pt = (codec_pts[:, codec_prompt_len:int((configs['max_seq_len'] - 1) / (configs['model']['frame_ratio'] + 1) * configs['model']['frame_ratio']), codebook_num] - configs['model']['codebook_dim'] * codebook_num)
        # mask = torch.full(codec_pt.shape, True).to(device)
        # outputs_current = torch.masked_select(outputs[:, :, :, codebook_num], mask.unsqueeze(2)).view((-1, configs['model']['codebook_dim']))
        # codec_pt = torch.masked_select(codec_pt, mask)
        loss += cross_entropy_loss(torch.transpose(outputs[:, :, :, codebook_num], 1, 2), codec_pt[:, :].detach()).item()
        del codec_pt
    del outputs
    print(loss)
    zq = audiodec.rx_encoder.lookup(torch.transpose(codec_prompt.squeeze(0), 0, 1))
    # zq = audiodec.rx_encoder.lookup(torch.transpose(codec_pts.squeeze(0), 0, 1))
    del codec_prompt
    del codec_pts
    y = audiodec.decoder.decode(zq)[:, :, :]
    torchaudio.save('/home/hltcoe/xli/ARTS/stream_voice/test.wav', y.squeeze(0).to('cpu'), 24000)
    del zq
    del y
    del batch
    breakpoint()
    