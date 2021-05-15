import sys
import random
import librosa
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from datasets.text.sequence import Sequence
from datasets.audio.stft import TacotronSTFT


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def load_wav_to_torch(full_path, sampling_rate=None):
    y, sr = librosa.core.load(full_path, sampling_rate)
    yt, _ = librosa.effects.trim(y)
    return torch.FloatTensor(yt.astype(np.float32)), sr


class Text2MelDataset(Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """

    def __init__(self, filepaths_and_text, hparams):
        # self.max_dataset_size = hparams.max_dataset_size
        self.filepaths_and_text = load_filepaths_and_text(filepaths_and_text)
        self.seq = Sequence(graphemes_or_phonemes=hparams.graphemes_or_phonemes,
                            use_phonemes=hparams.use_phonemes,
                            specials=hparams.specials,
                            punctuations=hparams.punctuations)
        self.sampling_rate = hparams.sampling_rate
        # self.max_wav_value = hparams.max_wav_value
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = TacotronSTFT(filter_length=hparams.filter_length,
                                 hop_length=hparams.hop_length,
                                 win_length=hparams.win_length,
                                 n_mel_channels=hparams.n_mel_channels,
                                 sampling_rate=hparams.sampling_rate,
                                 mel_fmin=hparams.mel_fmin,
                                 mel_fmax=hparams.mel_fmax)
        random.seed(2021)
        random.shuffle(self.filepaths_and_text)

    def get_spec_text_pair(self, filepath_and_text):
        # separate filename and text
        filepath, text = filepath_and_text
        text = self.get_text(text)
        spec = self.get_spec(filepath)
        return (text, spec)

    def get_spec(self, filename):
        if self.load_mel_from_disk:
            melspec = torch.from_numpy(np.load(filename))
        else:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            # audio_norm = audio / self.max_wav_value
            audio_norm = audio.unsqueeze(0)
            audio_norm = Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = melspec.squeeze(0)

        return melspec

    def get_text(self, text):
        text = self.seq.text_to_sequence(text)
        return text

    def __getitem__(self, index):
        return self.get_spec_text_pair(self.filepaths_and_text[index])

    def __len__(self):
        return len(self.filepaths_and_text)


class Text2MelDataLoader(DataLoader):
    def __init__(self, text2mel_dataset, hparams, \
                 shuffle=True, num_workers=0 if sys.platform.startswith('win') else 8, **kwargs):
        collate_fn = Text2MelCollate(n_frames_per_step=hparams.n_frames_per_step)
        super(Text2MelDataLoader, self).__init__(
                 dataset=text2mel_dataset, batch_size=hparams.batch_size,
                 shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, **kwargs)


class Text2MelCollate:
    """ Zero-pads model inputs and targets based on number of frames per step
    """

    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        batch_size = len(batch)
        # Right zero-pad all one-hot text sequences to max input length
        text_lengths, ids_sorted = \
            torch.LongTensor([len(x[0]) for x in batch]).sort(dim=0, descending=True)
        max_text_len = text_lengths[0]

        text_padded = torch.LongTensor(batch_size, max_text_len)
        text_padded.zero_()
        for i in range(len(ids_sorted)):
            text = batch[ids_sorted[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad melspec
        n_mel_channels = batch[0][1].size(0)
        max_spec_len = max([x[1].size(1) for x in batch])
        if max_spec_len % self.n_frames_per_step != 0:
            max_spec_len += self.n_frames_per_step - max_spec_len % self.n_frames_per_step
            assert max_spec_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        spec_padded = torch.FloatTensor(batch_size, n_mel_channels, max_spec_len)
        spec_padded.zero_()
        gate_padded = torch.FloatTensor(batch_size, max_spec_len)
        gate_padded.zero_()
        spec_lengths = torch.LongTensor(batch_size)
        for i in range(len(ids_sorted)):
            spec = batch[ids_sorted[i]][1]
            spec_padded[i, :, :spec.size(1)] = spec
            gate_padded[i, spec.size(1) - 1:] = 1
            spec_lengths[i] = spec.size(1)

        return text_padded, text_lengths, spec_padded, gate_padded, spec_lengths


if __name__ == '__main__':
    import config

    collate_fn = Text2MelCollate(config.n_frames_per_step)

    train_dataset = Text2MelDataset(config.train_files, config)
    print('len(train_dataset): ' + str(len(train_dataset)))

    valid_dataset = Text2MelDataset(config.valid_files, config)
    print('len(valid_dataset): ' + str(len(valid_dataset)))

    text, spec = valid_dataset[0]
    print('type(spec): ' + str(type(spec)))

    text_lengths = []
    spec_lengths = []

    for data in valid_dataset:
        text, spec = data
        text = valid_dataset.seq.sequence_to_text(text.numpy().tolist())
        text = ''.join(text)
        spec = spec.numpy()

        print('text: ' + str(text))
        print('spec.size: ' + str(spec.size))
        text_lengths.append(len(text))
        spec_lengths.append(spec.size)
        # print('np.mean(spec): ' + str(np.mean(spec)))
        # print('np.max(spec): ' + str(np.max(spec)))
        # print('np.min(spec): ' + str(np.min(spec)))

    print('np.mean(text_lengths): ' + str(np.mean(text_lengths)))
    print('np.mean(spec_lengths): ' + str(np.mean(spec_lengths)))

    train_loader = Text2MelDataLoader(train_dataset, config, shuffle=True)
    print('len(train_loader): ' + str(len(train_loader)))

    valid_loader = Text2MelDataLoader(valid_dataset, config, shuffle=False)
    print('len(valid_loader): ' + str(len(valid_loader)))

    batch = iter(valid_loader).next()
    print('type(spec): ' + str(type(batch)))
    print('batch[0].size(): ' + str(batch[0].size()))
    print('batch[1].size(): ' + str(batch[1].size()))
    print('batch[2].size(): ' + str(batch[2].size()))
    print('batch[3].size(): ' + str(batch[3].size()))
    print('batch[4].size(): ' + str(batch[4].size()))
