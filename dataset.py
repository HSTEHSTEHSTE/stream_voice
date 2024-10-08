import torch, torchaudio
from torch.utils.data import Dataset, DataLoader
import os, math

class Dataset(Dataset):
    def __init__(self, configs, sort = True, set_name = None):
        assert set_name is None or set_name in ['train', 'dev', 'test', 'wildcard']
        if set_name is None or set_name == 'train':
            sets = configs['data']['train_sets']
        elif set_name == 'dev':
            sets = configs['data']['dev_sets']
        elif set_name == 'test':
            sets = configs['data']['test_sets']
        elif set_name == 'wildcard':
            sets = configs['data']['wildcard']

        self.max_seqlen = configs['max_seq_len']
        self.sort = sort

        self.codec_path = configs['data']['codec_path']
        self.asr_path = configs['data']['asr_path']
        self.wav_path = configs['data']['wav_path']

        self.codec_size = configs['model']['codebook_num'] * configs['model']['codebook_dim']
        self.frame_ratio = configs['model']['frame_ratio']

        self.codebook_ids = configs['model']['codebook_ids']
        self.codebook_dim = configs['model']['codebook_dim']

        self.items = {}
        self.item_indices = []
        for data_set in sets:
            with open(os.path.join(configs['data']['base_path'], data_set + '.txt')) as meta_file:
                for line in meta_file:
                    line_items = line.strip().split(',')
                    self.items[line_items[0]] = {}
                    self.items[line_items[0]]['spk'] = line_items[1]
                    self.items[line_items[0]]['path'] = line_items[2]
                    self.item_indices.append(line_items[0])


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item_meta = self.items[self.item_indices[idx]].copy()
        item_meta['codec'] = torch.load(os.path.join(self.codec_path, item_meta['path'] + '.pt'), map_location = 'cpu').detach()
        if len(item_meta['codec'].shape) == 3:
            item_meta['codec'] = item_meta['codec'].squeeze(1)
            for codebook_num in self.codebook_ids:
                item_meta['codec'][codebook_num] += self.codebook_dim * codebook_num
        item_meta['asr_emb'] = torch.load(os.path.join(self.asr_path, item_meta['path'] + '.pt'), map_location = 'cpu')[:, :-7, :]
        item_meta['asr_emb'] = torch.nn.functional.interpolate(
            input = torch.transpose(item_meta['asr_emb'], 1, 2),
            size = math.ceil(item_meta['codec'].shape[1] / self.frame_ratio)
        )
        item_meta['asr_emb'] = torch.transpose(item_meta['asr_emb'], 1, 2).detach()
        item_meta['wav'] = torchaudio.load(os.path.join(self.wav_path, item_meta['path'] + '.wav'))[0].squeeze(0)
        return item_meta

    def collate_fn(self, batch):
        batch_len = len(batch)
        codec_pts = []
        codec_lens = []
        asr_emb_pts = []
        asr_emb_lens = []
        wavs = []
        wav_lens = []
        paths = []

        for index in range(batch_len):
            codec_pts.append(torch.transpose(batch[index]['codec'], 1, 0)) # [length, 8]
            codec_lens.append(batch[index]['codec'].shape[1])
            asr_emb_pts.append(batch[index]['asr_emb'].squeeze(0)) # [length, 384]
            asr_emb_lens.append(batch[index]['asr_emb'].shape[1])
            wavs.append(batch[index]['wav'])
            wav_lens.append(batch[index]['wav'].shape[0])
            paths.append(batch[index]['path'])
        codec_pts = torch.nn.utils.rnn.pad_sequence(
            sequences = codec_pts,
            batch_first = True,
            padding_value = self.codec_size
        )
        asr_emb_pts = torch.nn.utils.rnn.pad_sequence(
            sequences = asr_emb_pts,
            batch_first = True,
            padding_value = -1.
        )
        wavs = torch.nn.utils.rnn.pad_sequence(
            sequences = wavs,
            batch_first = True,
            padding_value = -999.
        )

        out = {
            "codec_pts": codec_pts,
            "codec_lens": torch.tensor(codec_lens),
            "asr_emb_pts": asr_emb_pts,
            "asr_emb_lens": torch.tensor(asr_emb_lens),
            "wavs": wavs,
            "wav_lens": torch.tensor(wav_lens),
            "paths": paths
        }
        return out


if __name__ == "__main__":
    # Test
    import yaml
    with open('config.yaml') as stream:
        configs = yaml.safe_load(stream)
    dataset = Dataset(configs, set_name = 'dev')

    print(len(dataset))
    dataset.__getitem__(1)
