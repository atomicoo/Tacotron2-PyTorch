import sys
import torch
sys.path.append('..')
from datasets.text.sequence import Sequence

seq = Sequence(graphemes_or_phonemes=list('abcdefghijklmnopqrstuvwxyz12345 '))
sequence = seq.text_to_sequence('bi4 xu1 shu4 li4 gong1 gong4 jiao1 tong1 you1 xian1 fa1 zhan3 de5 li3 nian4')
print(sequence)
