import torch
import torch.nn as nn


class Sequence(nn.Module):
    def __init__(self, graphemes_or_phonemes=[], use_phonemes=True,
                 specials=[], punctuations=[]):
        super(Sequence, self).__init__()
        self.phonemize = use_phonemes
        self.specials = specials
        self.graphemes_or_phonemes = graphemes_or_phonemes
        self.punctuations = punctuations
        self.units = self.specials + graphemes_or_phonemes + self.punctuations

        self.txt2idx = {txt: idx for idx, txt in enumerate(self.units)}
        self.idx2txt = {idx: txt for idx, txt in enumerate(self.units)}

    def text_to_sequence(self, text):
        # text = chinese_cleaners(text)
        sequence = torch.IntTensor([self.txt2idx[ch] for ch in text])
        return sequence

    def sequence_to_text(self, sequence):
        text = [self.idx2txt[idx] for idx in sequence]
        return text

