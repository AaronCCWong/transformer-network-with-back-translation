import torch
from copy import deepcopy


CONSTANTS = {
    'pad': '<pad>',
    'start': '<s>',
    'end': '</s>'
}


def clone_layer(layer, num_clones):
    return [deepcopy(layer) for i in range(num_clones)]


def generate_word_dict(pathToEnglishData, pathToFrenchData):
    word_dict = {}
    with open('data/train/giga-fren.release2.fixed.en', 'r') as file:
        line = file.readline()
        while line:
            # Process data
            break


def padding_mask(seq, src_vocab):
    rows, cols = seq.size()
    return (seq != src_vocab.stoi[CONSTANTS['pad']]).unsqueeze(-2)


def subsequent_mask(seq):
    rows, cols = seq.size()
    mask = torch.triu(torch.ones((rows, cols), dtype=torch.uint8), diagonal=1)
    return (mask == 0).unsqueeze(-2)


def tokenize(spacy):
    def tokenize_text(text):
        return [token.text for token in spacy.tokenizer(text)]
    return tokenize_text


class AverageMeter(object):
    """Taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
