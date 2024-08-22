import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import yaml
import shutil
from pathlib import Path
from tqdm import tqdm
import torch, torchaudio
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from dataset import Dataset
from model.lm import StreamVoice
from model.vocoder_models.models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint, get_checkpoints, update_checkpoints, get_param_num, mel_spectrogram

torch.backends.cudnn.benchmark = True


def advance_one_step(configs, inference_sampler, codec_prompt, codec_extension, codec_extra_pad, model, asr_emb_prompt, current_codec_pos):
    
    codec_prompt_ext = torch.cat([codec_prompt, codec_extension.detach()], dim = 1).detach()
    codec_prompt_ext = torch.cat([codec_prompt_ext, codec_extra_pad.detach()], dim = 1).detach()

    output = model(codec_prompt_ext.detach(), asr_emb_prompt.detach())
    del codec_prompt_ext
    output = output[:, :-1, :, :]
    output = output.view((output.shape[0], int(output.shape[1] / (configs['model']['frame_ratio'] + 1)), configs['model']['frame_ratio'] + 1, output.shape[2], output.shape[3]))
    output = output[:, :, :-1, :, :]
    output = torch.reshape(output, (output.shape[0], output.shape[1] * output.shape[2], output.shape[3], output.shape[4]))
    if configs['training']['top_k'] == 1 or configs['training']['temperature'] == 0:
        next_tokens = torch.argmax(output[:, current_codec_pos, :, :], dim = 1).unsqueeze(1)
    else:
        next_token_candidates = torch.topk(output[:, current_codec_pos, :, :], k = configs['training']['top_k'], dim = 1)
        next_token_probs = inference_sampler(torch.div(next_token_candidates.values, configs['training']['temperature']))
        next_token_probs = next_token_probs.transpose(1, 2)
        next_token_probs = next_token_probs.reshape(next_token_probs.shape[0] * next_token_probs.shape[1], next_token_probs.shape[2])
        next_token_candidate_indices = torch.multinomial(next_token_probs, 1).squeeze(1)
        next_token_candidates = next_token_candidates.indices
        next_token_candidates = next_token_candidates.transpose(1, 2)
        next_token_candidates = next_token_candidates.reshape(next_token_candidates.shape[0] * next_token_candidates.shape[1], next_token_candidates.shape[2])
        next_tokens = torch.diagonal(torch.index_select(next_token_candidates, dim = 1, index = next_token_candidate_indices), dim1 = 0, dim2 = 1).view(output.shape[0], -1).unsqueeze(1)
        del next_token_candidates
        del next_token_probs

    for codebook_num in range(configs['model']['codebook_num']):
        next_tokens[:, :, codebook_num] += codebook_num * configs['model']['codebook_dim']
    codec_prompt = torch.cat([codec_prompt, next_tokens], dim = 1)
    del output
    return codec_prompt, next_tokens


def train(configs):
    # Verify experiment folder
    exp_path = os.path.join(configs['exp_path'], configs['exp_name'])
    checkpoint_path = os.path.join(exp_path, 'ckpt')
    assert os.path.isdir(checkpoint_path)
    checkpoint_hifigan_path = os.path.join(exp_path, 'ckpt_hifigan')
    Path(checkpoint_hifigan_path).mkdir(parents = True, exist_ok = True)
    logs_hifigan_path = os.path.join(exp_path, 'logs_hifigan')
    Path(logs_hifigan_path).mkdir(parents = True, exist_ok = True)
    ## Copy configs into experiment folder
    shutil.copyfile('config_hifigan.yaml', os.path.join(exp_path, 'config_hifigan.yaml'))
    ## Reload configs
    with open(os.path.join(exp_path, 'config_hifigan.yaml')) as stream:
        configs = yaml.safe_load(stream)
    print(configs, flush = True)

    # verify device settings
    torch.manual_seed(configs['training']['random_seed'])
    if configs['device'] == 'cuda':
        assert "CUDA_VISIBLE_DEVICES" in os.environ and len(os.environ["CUDA_VISIBLE_DEVICES"]) > 0
        torch.cuda.manual_seed(configs['training']['random_seed'])
        print('CUDA enabled', flush = True)
    print('CUDA devices: ', os.environ["CUDA_VISIBLE_DEVICES"], flush = True)
    device = torch.device(configs['device'])

    # Define Codec Model
    model = torch.nn.DataParallel(StreamVoice(configs))
    codec_prompt_len = int(configs['model']['prompt_len'] / (configs['model']['frame_ratio'] + 1) * configs['model']['frame_ratio'])
    ## Load checkpoint if possible
    current_step = 0
    checkpoints = get_checkpoints(checkpoint_path)
    restore_checkpoint = -1
    if configs['training']['restore_step'] in checkpoints.keys() or (configs['training']['restore_step'] == 'infer' and len(checkpoints.keys()) > 0):
        if configs['training']['restore_step'] in checkpoints.keys():
            restore_checkpoint = configs['training']['restore_step']
        if configs['training']['restore_step'] == 'infer':
            restore_checkpoint = max(checkpoints.keys())
        checkpoint = torch.load(os.path.join(checkpoint_path, 'checkpoint_{}.pt'.format(restore_checkpoint)))
        model.load_state_dict(checkpoint['model'])
        del checkpoint
        print("\n---Model Restored at Step {}---\n".format(restore_checkpoint))
        current_step = restore_checkpoint
    model = model.to(device)

    # Define Hifigan Model
    generator = Generator(configs).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    print('Number of Hifigan model parameters: ', get_param_num(generator), flush = True)
    print('Number of discriminator model parameters: ', get_param_num(mpd) + get_param_num(msd), flush = True)

    ## Load checkpoint if possible
    cp_g = scan_checkpoint(checkpoint_hifigan_path, 'g_')
    cp_do = scan_checkpoint(checkpoint_hifigan_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    # Define optimizer scheduler
    optim_g = torch.optim.AdamW(generator.parameters(), 
        configs['training']['optim']['learning_rate'], 
        betas = configs['training']['optim']['betas']
    )
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
        configs['training']['optim']['learning_rate'], 
        betas = configs['training']['optim']['betas']
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, 
        gamma = configs['training']['optim']['lr_decay'], 
        last_epoch = last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, 
        gamma = configs['training']['optim']['lr_decay'], 
        last_epoch = last_epoch
    )

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
        batch_size = 1, 
        shuffle = False,
        collate_fn = dataset.collate_fn, 
        drop_last = False, 
        num_workers = 16
    )
    print('Length of validation dataset: ', len(validation_dataset), flush = True)

    if configs['training']['top_k'] > 1 and configs['training']['temperature'] > 0:
        inference_sampler = torch.nn.Softmax(dim = 1)

    model.eval()
    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), configs['training']['epoch']):
        start = time.time()
        print("Epoch: {}".format(epoch))

        for batch_index, batch in enumerate(loader):
            start_b = time.time()

            wavs = batch['wavs']

            with torch.no_grad():
                codec_pts = batch['codec_pts'].detach().to(device)
                asr_emb_pts = batch['asr_emb_pts'].detach().to(device)

                if codec_pts.shape[1] > codec_prompt_len:
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
                            _, next_tokens = advance_one_step(configs, inference_sampler, codec_prompt_in, codec_extension, codec_extra_pad, model, asr_emb_prompt, current_codec_pos)
                            codec_prompt = torch.cat([codec_prompt, next_tokens], dim = 1)
                            del next_tokens
                            del codec_prompt_in
                            current_codec_pos += 1
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
                                _, next_tokens = advance_one_step(configs, inference_sampler, codec_prompt_in, codec_extension, codec_extra_pad, model, asr_emb_prompt, current_codec_pos)
                                codec_prompt = torch.cat([codec_prompt, next_tokens], dim = 1)
                                del next_tokens
                                del codec_prompt_in
                                current_codec_pos += 1
                                codec_prompt_in = codec_prompt
                                del codec_extension
                    del codec_extra_pad
                del asr_emb_pts

            wav_len = codec_prompt.shape[1] * 256
            wavs = wavs[:, :wav_len].to(device)
            wavs = wavs.unsqueeze(1)
            mels = mel_spectrogram(
                wavs.squeeze(1), 
                configs['mel']['n_fft'], 
                configs['mel']['num_mels'], 
                configs['data']['sampling_rate'], 
                configs['mel']['hop_size'], 
                configs['mel']['win_size'], 
                configs['mel']['fmin'], 
                configs['mel']['fmax_for_loss']
            )

            wavs_prediction = generator(codec_prompt)
            mels_prediction = mel_spectrogram(
                wavs_prediction.squeeze(1), 
                configs['mel']['n_fft'], 
                configs['mel']['num_mels'], 
                configs['data']['sampling_rate'], 
                configs['mel']['hop_size'], 
                configs['mel']['win_size'], 
                configs['mel']['fmin'], 
                configs['mel']['fmax_for_loss']
            )

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(wavs, wavs_prediction.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(wavs, wavs_prediction.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(mels, mels_prediction) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(wavs, wavs_prediction)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(wavs, wavs_prediction)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            optim_g.step()

            # STDOUT logging
            if steps % configs['training']['report_every'] == 0:
                with torch.no_grad():
                    mel_error = F.l1_loss(mels, mels_prediction).item()

                print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.format
                    (
                        steps, 
                        loss_gen_all, 
                        mel_error, 
                        time.time() - start_b
                    ), flush = True)

            # checkpointing
            if steps % configs['training']['save_every'] == 0 and steps != 0:
                save_path = "{}/g_{:08d}".format(checkpoint_hifigan_path, steps)
                save_checkpoint(
                    save_path,
                    {
                        'generator': generator.state_dict()
                    }
                )
                save_path = "{}/do_{:08d}".format(checkpoint_hifigan_path, steps)
                save_checkpoint(
                    save_path, 
                    {
                        'mpd': mpd.state_dict(),
                        'msd':  msd.state_dict(),
                        'optim_g': optim_g.state_dict(), 
                        'optim_d': optim_d.state_dict(), 
                        'steps': steps,
                        'epoch': epoch
                    }
                )

            # Validation
            if steps % configs['training']['valid_every'] == 0:  # and steps != 0:
                generator.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0
                with torch.no_grad():
                    for validation_batch_index, validation_batch in tqdm(enumerate(validation_loader), total = min(configs['training']['valid_length_limit'], len(validation_loader))):
                        if validation_batch_index >= configs['training']['valid_length_limit']:
                            break

                        wavs = validation_batch['wavs']
                        wav_lens = validation_batch['wav_lens']

                        codec_pts = validation_batch['codec_pts'].detach().to(device)
                        asr_emb_pts = validation_batch['asr_emb_pts'].detach().to(device)

                        if codec_pts.shape[1] > codec_prompt_len:
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
                                    _, next_tokens = advance_one_step(configs, inference_sampler, codec_prompt_in, codec_extension, codec_extra_pad, model, asr_emb_prompt, current_codec_pos)
                                    codec_prompt = torch.cat([codec_prompt, next_tokens], dim = 1)
                                    del next_tokens
                                    del codec_prompt_in
                                    current_codec_pos += 1
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
                                        _, next_tokens = advance_one_step(configs, inference_sampler, codec_prompt_in, codec_extension, codec_extra_pad, model, asr_emb_prompt, current_codec_pos)
                                        codec_prompt = torch.cat([codec_prompt, next_tokens], dim = 1)
                                        del next_tokens
                                        del codec_prompt_in
                                        current_codec_pos += 1
                                        codec_prompt_in = codec_prompt
                                        del codec_extension
                            del codec_extra_pad
                        del asr_emb_pts


                        wav_len = codec_prompt.shape[1] * 256
                        wavs = wavs[:, :wav_len].to(device)
                        wavs = wavs.unsqueeze(1)
                        mels = mel_spectrogram(
                            wavs.squeeze(1), 
                            configs['mel']['n_fft'], 
                            configs['mel']['num_mels'], 
                            configs['data']['sampling_rate'], 
                            configs['mel']['hop_size'], 
                            configs['mel']['win_size'], 
                            configs['mel']['fmin'], 
                            configs['mel']['fmax_for_loss']
                        )

                        wavs_prediction = generator(codec_prompt)
                        mels_prediction = mel_spectrogram(
                            wavs_prediction.squeeze(1), 
                            configs['mel']['n_fft'], 
                            configs['mel']['num_mels'], 
                            configs['data']['sampling_rate'], 
                            configs['mel']['hop_size'], 
                            configs['mel']['win_size'], 
                            configs['mel']['fmin'], 
                            configs['mel']['fmax_for_loss']
                        )
                        val_err_tot += F.l1_loss(mels, mels_prediction).item()

                        current_logs_hifigan_path = os.path.join(logs_hifigan_path, 'step_' + str(steps))
                        Path(current_logs_hifigan_path).mkdir(parents = True, exist_ok = True)
                        if validation_batch_index <= 4:
                            if steps == 0:
                                torchaudio.save(
                                    os.path.join(
                                        current_logs_hifigan_path, 
                                        'wav_' + str(validation_batch_index) + '.wav'
                                    ),
                                    wavs[0, :, :wav_lens[0]].to('cpu'),
                                    configs['data']['sampling_rate']
                                )
                                plot_spectrogram(mels[0].to('cpu')).savefig(
                                    os.path.join(
                                        current_logs_hifigan_path, 
                                        'mel_' + str(validation_batch_index) + '.png'
                                    ),
                                )

                            torchaudio.save(
                                os.path.join(
                                    current_logs_hifigan_path, 
                                    'wav_pred_' + str(validation_batch_index) + '.wav'
                                ),
                                wavs_prediction[0, :, :wav_lens[0]].to('cpu'),
                                configs['data']['sampling_rate']
                            )
                            mel_prediction = mel_spectrogram(
                                wavs_prediction[0, :, :wav_lens[0]], 
                                configs['mel']['n_fft'], 
                                configs['mel']['num_mels'], 
                                configs['data']['sampling_rate'], 
                                configs['mel']['hop_size'], 
                                configs['mel']['win_size'], 
                                configs['mel']['fmin'], 
                                configs['mel']['fmax']
                            )
                            plot_spectrogram(mel_prediction[0].to('cpu')).savefig(
                                os.path.join(
                                    current_logs_hifigan_path, 
                                    'mel_pred_' + str(validation_batch_index) + '.png'
                                ),
                            )

                    val_err = val_err_tot / (validation_batch_index + 1)
                    print('Steps : {:d}, Validation Mel-Spec. Error : {:4.3f}'.format
                    (
                        steps, 
                        mel_error, 
                    ), flush = True)

                generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)), flush = True)


def main():
    print('Initializing Training Process..', flush = True)

    with open('config_hifigan.yaml') as stream:
        configs = yaml.safe_load(stream)

    train(configs)


if __name__ == '__main__':
    main()
