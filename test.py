import yaml, os, shutil
from pathlib import Path
import torch, torchaudio
from dataset import Dataset
from model.lm import StreamVoice
from tqdm import tqdm
from model.audiodec import AudioDec, assign_model
from utils import advance_one_step

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', '-e', type = str, required = True)
parser.add_argument('--restore-step', '-r', type = int, default = 0)
parser.add_argument('--top-k', '-k', type = int, default = 1)
parser.add_argument('--temperature', '-t', type = float, default = 0.)
parser.add_argument('--dump_path', '-d', type = str, default = '')
parser.add_argument('--dump_flat', '-f', type = int, default = 1)


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
    batch_size = 4,
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

prompt = torch.load('temp.pt').to(device)

if args.top_k > 1 and args.temperature > 0:
    inference_sampler = torch.nn.Softmax(dim = 1)

codec_prompt_len = int(configs['model']['prompt_len'] / (configs['model']['frame_ratio'] + 1) * configs['model']['frame_ratio'])

loss_weights_total = 0.
for codebook_index in range(configs['model']['codebook_num']):
    if codebook_index in configs['model']['codebook_ids']:
        loss_weights_total += configs['model']['codebook_weights'][codebook_index]
loss_weights = [x / loss_weights_total for x in configs['model']['codebook_weights']]

with torch.no_grad():
    for batch_index, batch in tqdm(enumerate(loader), total = len(loader)):
        codec_pts = batch['codec_pts'].detach().to(device)
        asr_emb_pts = batch['asr_emb_pts'].detach().to(device)

        if codec_pts.shape[1] > codec_prompt_len:
            outputs = []
            codec_prompt = codec_pts[:, :codec_prompt_len, :]
            # codec_prompt = prompt
            codec_prompt_in = codec_pts[:, :codec_prompt_len, :]
            asr_emb_prompt_len = int(configs['model']['prompt_len'] / (configs['model']['frame_ratio'] + 1))
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
                    _, outputs, next_tokens = advance_one_step(codec_prompt_in, codec_extension, codec_extra_pad, model, asr_emb_prompt, current_codec_pos, outputs, codec_pts, configs)
                    codec_prompt = torch.cat([codec_prompt, next_tokens], dim = 1)
                    del next_tokens
                    del codec_prompt_in
                    current_codec_pos += 1
                    # codec_prompt_in = codec_pts[:, :current_codec_pos, :]
                    codec_prompt_in = codec_prompt
                    del codec_extension
                else:
                    if int(codec_prompt.shape[1] / configs['model']['frame_ratio']) == current_asr_emb_pos:
                        # next token is asr_emb, skip forward
                        current_asr_emb_pos += 1
                        del asr_emb_prompt
                        asr_emb_prompt = asr_emb_pts[:, :current_asr_emb_pos, :]
                    else:
                        codec_extension = torch.full(
                            size = (codec_prompt.shape[0], configs['model']['frame_ratio'], codec_prompt.shape[2]), 
                            fill_value = configs['model']['codebook_num'] * configs['model']['codebook_dim']
                        ).to(device)
                        _, outputs, next_tokens = advance_one_step(codec_prompt_in, codec_extension, codec_extra_pad, model, asr_emb_prompt, current_codec_pos, outputs, codec_pts, configs)
                        codec_prompt = torch.cat([codec_prompt, next_tokens], dim = 1)
                        del next_tokens
                        del codec_prompt_in
                        current_codec_pos += 1
                        # codec_prompt_in = codec_pts[:, :current_codec_pos, :]
                        codec_prompt_in = codec_prompt
                        del codec_extension
            del codec_extra_pad
        del asr_emb_pts
        loss = 0
        total_loss = 0
        losses = {}
        total = 0.
        correct = 0.
        outputs = torch.stack(outputs, dim = 1).to(device)
        for codebook_num in range(configs['model']['codebook_num']):
            if codebook_num in configs['model']['codebook_ids']:
                codec_pt = (codec_pts[:, codec_prompt_len:int((configs['max_seq_len'] - 1) / (configs['model']['frame_ratio'] + 1) * configs['model']['frame_ratio']), codebook_num] - configs['model']['codebook_dim'] * codebook_num)
                
                mask = codec_pt != dataset.codec_size - configs['model']['codebook_dim'] * codebook_num
                codec_pt = torch.masked_select(codec_pt, mask)
                output_codec = torch.masked_select(outputs[:, :, :, codebook_num], mask.unsqueeze(2)).view((-1, configs['model']['codebook_dim']))
                
                current_loss = cross_entropy_loss(output_codec, codec_pt.detach()).item()
                loss += current_loss * loss_weights[codebook_num]
                total_loss += current_loss
                if codebook_num not in losses:
                    losses[codebook_num] = current_loss
                else:
                    losses[codebook_num] += current_loss
                
                # mask = torch.full(codec_pt.shape, True).to(device)
                # outputs_current = torch.masked_select(outputs[:, :, :, codebook_num], mask.unsqueeze(2)).view((-1, configs['model']['codebook_dim']))
                # codec_pt = torch.masked_select(codec_pt, mask)
                
                # loss += cross_entropy_loss(torch.transpose(outputs[:, :, :, codebook_num], 1, 2), codec_pt[:, :].detach()).item()
                correct += torch.sum(torch.eq(torch.topk(output_codec, k = 10, dim = 1).indices, codec_pt.unsqueeze(1))).item()
                total += codec_pt.shape[0]
                del codec_pt
        del outputs
        print('Test weighted loss: ', loss)
        print('Test total loss: ', total_loss)
        for codebook_index in configs['model']['codebook_ids']:
            print('Test loss for codebook: ', codebook_index, " : ", losses[codebook_index])
        print('Test top 10 accuracy: ', correct / total, flush = True)
        # codec_prompt[:, :codec_prompt_len, :] = codec_pts[:, :codec_prompt_len, :]
        del codec_pts
        
        for i in range(codec_prompt.shape[0]):
            zq = audiodec.rx_encoder.lookup(torch.transpose(codec_prompt[i, :min(codec_prompt.shape[1], batch['codec_lens'][i]), :], 0, 1))
            # zq = audiodec.rx_encoder.lookup(torch.transpose(codec_pts.squeeze(0), 0, 1))
            y = audiodec.decoder.decode(zq)[:, :, :]
            if len(args.dump_path) == 0:
                torchaudio.save(
                    '/home/hltcoe/xli/ARTS/stream_voice/test.wav', 
                    y.squeeze(0).to('cpu'), 
                    configs['data']['sampling_rate']
                )
                breakpoint()
            else:
                if args.dump_flat == 0:
                    target_path = os.path.join(args.dump_path, batch['paths'][i] + '.wav')
                else:
                    path_elements = batch['paths'][i].split('/')
                    wav_name = path_elements[0] + '/' + path_elements[-1] + '.wav'
                    target_path = os.path.join(args.dump_path, wav_name)
                Path('/'.join(target_path.split('/')[:-1])).mkdir(parents = True, exist_ok = True)
                torchaudio.save(target_path, y.squeeze(0).to('cpu'), configs['data']['sampling_rate'])
            del y
            del zq
        del codec_prompt
        del batch
    