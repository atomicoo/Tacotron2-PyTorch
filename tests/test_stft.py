import sys
import torch
import librosa
sys.path.append('..')
from datasets.audio.stft import TacotronSTFT
from utils.plot import plot_spectrogram

fullpath = '../audios/LJ001-0007.wav'

filter_length = 1024
hop_length = 256
win_length = 1024
n_mel_channels = 80
sampling_rate = 22050
mel_fmin = 0.0 # 80.0
mel_fmax = 8000.0 # 7600.0

stft = TacotronSTFT(filter_length=filter_length,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    sampling_rate=sampling_rate,
                    mel_fmin=mel_fmin,
                    mel_fmax=mel_fmax)

wav, sr = librosa.load(fullpath, sr=None)

assert sr == sampling_rate

wav = torch.from_numpy(wav).unsqueeze(0)
mel = stft.mel_spectrogram(wav).squeeze(0).t()

print(mel.size())
plot_spectrogram(pred_spectrogram=mel, save_img=True, path='test.png')
