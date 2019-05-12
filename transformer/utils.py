import spacy
import torch
import torch.nn.functional as F
from copy import deepcopy
from itertools import chain
from torchtext import data, datasets


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


def build_dataset(args):
    print('Loading spacy language models...')
    src = data.Field(tokenize=get_tokenizer(args.src_language), lower=True, pad_token=CONSTANTS['pad'])
    tgt = data.Field(tokenize=get_tokenizer(args.tgt_language),
                     lower=True,
                     init_token=CONSTANTS['start'],
                     pad_token=CONSTANTS['pad'],
                     eos_token=CONSTANTS['end'])
    print('Finished loading spacy language models.')

    print('Loading data splits...')
    multi30k_train_gen, multi30k_val_gen, _ = datasets.Multi30k.splits(exts=(build_file_extension(args.src_language), build_file_extension(args.tgt_language)),
                                                         fields=(('src', src), ('tgt', tgt)),
                                                         filter_pred=lambda x: len(vars(x)['src']) <= args.max_seq_length and len(vars(x)['tgt']) <= args.max_seq_length)

    iwslt_train_gen, iwslt_val_gen, _ = datasets.IWSLT.splits(exts=(build_file_extension(args.src_language), build_file_extension(args.tgt_language)),
                                                              fields=(('src', src), ('tgt', tgt)),
                                                              filter_pred=lambda x: len(vars(x)['src']) <= args.max_seq_length and len(vars(x)['tgt']) <= args.max_seq_length)

    backt_gen  = data.TabularDataset.splits(path='data', train='train.csv', format='csv',
                                            fields=(('src', src), ('tgt', tgt)),
                                            filter_pred=lambda x: len(vars(x)['src']) <= args.max_seq_length and len(vars(x)['tgt']) <= args.max_seq_length)[0]

    training_examples = multi30k_train_gen.examples + iwslt_train_gen.examples + backt_gen.examples
    validation_examples = multi30k_val_gen.examples + iwslt_val_gen.examples

    training_data = data.Dataset(training_examples, multi30k_train_gen.fields)
    validation_data = data.Dataset(validation_examples, multi30k_train_gen.fields)
    print('Finished loading data splits.')

    print('Building vocabulary...')
    train_gen_src = chain(multi30k_train_gen.src, iwslt_train_gen.src)
    train_gen_tgt = chain(multi30k_train_gen.tgt, iwslt_train_gen.tgt)

    src.build_vocab(train_gen_src, min_freq=args.min_word_freq)
    tgt.build_vocab(train_gen_tgt, min_freq=args.min_word_freq)
    print('Finished building vocabulary.')

    train_iterator, val_iterator, _ = data.BucketIterator.splits((training_data, validation_data, _),
                                                                  sort_key=lambda x: len(x.src),
                                                                  batch_sizes=(args.batch_size, args.batch_size, args.batch_size))


    return src, tgt, train_iterator, val_iterator


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
    return (seq != src_vocab.stoi[CONSTANTS['pad']]).unsqueeze(-2)

import numpy as np

def subsequent_mask(seq):
    rows, cols = seq.size()
    mask = np.triu(np.ones((1, cols, cols)), k=1)
    # mask = torch.triu(torch.ones((rows, cols), dtype=torch.uint8), diagonal=1)
    return torch.from_numpy(mask) == 0


def tokenize(spacy):
    def tokenize_text(text):
        return [token.text for token in spacy.tokenizer(text)]
    return tokenize_text
