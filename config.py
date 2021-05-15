import torch
import os.path as osp

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

meta_file = osp.join('data', 'BZNSYP', 'ProsodyLabeling', '000001-010000.txt')
wave_folder = osp.join('data', 'BZNSYP', 'Wave')

num_train = 9900
num_valid = 100

################################
# Experiment Parameters        #
################################
experiment = 'exp1'
logdir = osp.join('.', 'logdir')
epochs = 500
iters_per_checkpoint = 1000
seed = 2021
dynamic_loss_scaling = True
fp16_run = False
distributed_run = False

################################
# Data Parameters             #
################################
load_mel_from_disk = False
train_files = osp.join('data', 'filelists', 'bznsyp_audio_text_train_filelist.txt')
valid_files = osp.join('data', 'filelists', 'bznsyp_audio_text_valid_filelist.txt')

################################
# Text Parameters              #
################################
use_phonemes = False
graphemes_or_phonemes = list("abcdefghijklmnopqrstuvwxyz12345")
specials = list([])  # list(["<pad>", "<unk>"])
punctuations = list([".", ",", "?", "!", " ", "-"])

################################
# Audio Parameters             #
################################
max_wav_value = 32768.0
sampling_rate = 48000  # 22050
filter_length = 1024
hop_length = 256
win_length = 1024
n_mel_channels = 80
mel_fmin = 0.0
mel_fmax = 8000.0

################################
# Model Parameters             #
################################
n_symbols = len(graphemes_or_phonemes+specials+punctuations)
symbols_embedding_dim = 512

# Encoder parameters
encoder_kernel_size = 5
encoder_n_convolutions = 3
encoder_embedding_dim = 512

# Decoder parameters
n_frames_per_step = 1  # currently only 1 is supported
decoder_rnn_dim = 1024
prenet_dim = 256
max_decoder_steps = 1000
gate_threshold = 0.5
p_attention_dropout = 0.1
p_decoder_dropout = 0.1

# Attention parameters
attention_rnn_dim = 1024
attention_dim = 128

# Location Layer parameters
attention_location_n_filters = 32
attention_location_kernel_size = 31

# Mel-post processing network parameters
postnet_embedding_dim = 512
postnet_kernel_size = 5
postnet_n_convolutions = 5

################################
# Optimization Hyperparameters #
################################
learning_rate = 1e-3
weight_decay = 1e-6
batch_size = 64
mask_padding = True  # set model's padded outputs to padded values
