import spacy
import torch
import torch.nn.functional as F
from copy import deepcopy


CONSTANTS = {
    'pad': '<pad>',
    'start': '<s>',
    'end': '</s>'
}


LANGUAGES = {
    'ENGLISH': 'en',
    'GERMAN': 'de'
}


def build_file_extension(language):
    return '.' + language


def cal_performance(out, labels, tgt_vocab):
    loss = F.cross_entropy(out.view(-1, out.size(-1)), labels,
                           ignore_index=tgt_vocab.stoi[CONSTANTS['pad']],
                           reduction='sum')

    pred = out.max(2)[1].view(-1)
    labels = labels.contiguous().view(-1)
    non_pad_mask = labels.ne(tgt_vocab.stoi[CONSTANTS['pad']])
    n_correct = pred.eq(labels)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def clone_layer(layer, num_clones):
    return [deepcopy(layer) for i in range(num_clones)]


def get_tokenizer(language):
    if language == LANGUAGES['ENGLISH']:
        return tokenize(spacy.load('en_core_web_lg'))
    elif language == LANGUAGES['GERMAN']:
        return tokenize(spacy.load('de_core_news_sm'))
    else:
            raise Exception('Language provided is not supported')


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
