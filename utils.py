import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt
from librosa.filters import mel as librosa_mel_fn

def get_checkpoints(checkpoint_path):
    checkpoint_files = [f for dp, dn, filenames in os.walk(checkpoint_path) for f in filenames if os.path.splitext(f)[-1] == '.pt']
    checkpoints = {}
    for checkpoint_file in checkpoint_files:
        checkpoints[int(checkpoint_file[11:-3])] = checkpoint_file
    return checkpoints

def update_checkpoints(checkpoint_path, checkpoints, keep_checkpoints):
    checkpoint_steps = list(checkpoints.keys())
    checkpoint_steps.sort()
    checkpoint_steps = checkpoint_steps[:- keep_checkpoints]
    for checkpoint_step in checkpoint_steps:
        os.remove(os.path.join(checkpoint_path, checkpoints[checkpoint_step]))
        checkpoints.pop(checkpoint_step)
    return checkpoints


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # if torch.min(y) < -1.:
    #     print('min value is ', torch.min(y))
    # if torch.max(y) > 1.:
    #     print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr = sampling_rate, n_fft = n_fft, n_mels = num_mels, fmin = fmin, fmax = fmax)
        mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(
        y, 
        n_fft, 
        hop_length = hop_size, 
        win_length = win_size, 
        window = hann_window[str(y.device)],
        center = center, 
        pad_mode = 'reflect', 
        normalized = False, 
        onesided = True,
        return_complex = True
    )

    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def advance_one_step(
    codec_prompt, 
    codec_extension, 
    codec_extra_pad, 
    model, 
    asr_emb_prompt, 
    current_codec_pos, 
    outputs, 
    codec_pts,
    configs
):
    codec_prompt_len = configs['model']['codec_prompt_len']
    codec_prompt_ext = torch.cat([codec_prompt, codec_extension.detach()], dim = 1).detach()
    codec_prompt_ext = torch.cat([codec_prompt_ext, codec_extra_pad.detach()], dim = 1).detach()

    output, next_tokens = model(codec_prompt_ext.detach(), asr_emb_prompt.detach())
    del codec_prompt_ext
    output = output[:, codec_prompt_len:-1, :, :]
    output = output.view((output.shape[0], int(output.shape[1] / (configs['model']['frame_ratio'] + 1)), configs['model']['frame_ratio'] + 1, output.shape[2], output.shape[3]))
    output = output[:, :, :-1, :, :]
    output = torch.reshape(output, (output.shape[0], output.shape[1] * output.shape[2], output.shape[3], output.shape[4]))
    outputs.append(output[:, current_codec_pos - codec_prompt_len, :, :].detach().to('cpu'))

    next_tokens = next_tokens[:, codec_prompt_len:-1, :]
    next_tokens = next_tokens.view((next_tokens.shape[0], int(next_tokens.shape[1] / (configs['model']['frame_ratio'] + 1)), configs['model']['frame_ratio'] + 1, next_tokens.shape[2]))
    next_tokens = next_tokens[:, :, :-1, :]
    next_tokens = torch.reshape(next_tokens, (next_tokens.shape[0], next_tokens.shape[1] * next_tokens.shape[2], next_tokens.shape[3]))
    next_tokens = next_tokens[:, current_codec_pos - codec_prompt_len, :].unsqueeze(1)

    for codebook_num in range(configs['model']['codebook_num']):
        if codebook_num not in configs['model']['codebook_ids']:
            next_tokens[:, :, codebook_num] = codec_pts[:, current_codec_pos, codebook_num].unsqueeze(1)
    codec_prompt = torch.cat([codec_prompt, next_tokens], dim = 1)
    del output
    return codec_prompt, outputs, next_tokens