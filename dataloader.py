import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Mel2Samp(Dataset):
    def __init__(self, data_path, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_pad_val):
        self.wav_list = glob.glob(os.path.join(data_path, '**', 'quant', '*.npy'), recursive=True)
        self.mel_list = glob.glob(os.path.join(data_path, '**', 'gta', '*.npy'), recursive=True)
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.mel_pad_val = mel_pad_val
        self.mel_segment_length = segment_length // hop_length

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        mel = np.load(self.mel_list[idx])
        audio = np.load(self.wav_list[idx])

        if len(audio) < self.segment_length:
            mel = np.pad(mel, ((0, 0), (0, self.mel_segment_length - mel.shape[1])), mode='constant', constant_values=self.mel_pad_val)
            audio = np.pad(audio, (0, self.segment_length - len(audio)), mode='constant', constant_values=0.0)

        audio = torch.from_numpy(audio)
        mel = torch.from_numpy(mel)
        assert(self.hop_length * mel.size(1) == len(audio))

        max_mel_start = mel.size(1) - self.mel_segment_length
        mel_start = random.randint(0, max_mel_start)
        mel_end = mel_start + self.mel_segment_length
        mel = mel[:, mel_start:mel_end]

        audio_start = mel_start * self.hop_length
        audio = audio[audio_start:audio_start+self.segment_length]

        return mel, audio
