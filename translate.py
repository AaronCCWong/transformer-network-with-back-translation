import argparse
import math
import spacy
import torch
import torchtext
from torchtext import data, datasets

from transformer.transformer import Transformer
from transformer.utils import CONSTANTS, padding_mask, tokenize

def beam_search(model, src, src_mask, beam_size=3):
    for seq in src:
        encoded_seq = model.src_embedding(seq) * math.sqrt(model.d_model)
        encoded_seq = model.positional_encoder1(encoded_seq)
        encoded_seq = model.encoder(encoded_seq, src_mask)
        


def translate(model, dataset, src_vocab, tgt_vocab, args):
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataset):
            device = args.device
            src = batch.src.transpose(0, 1).to(device)
            src_mask = padding_mask(src, src_vocab)

            beam_search(model, src, args.beam_size)


def run(args):
    print('Loading spacy language models...')
    spacy_en = spacy.load('en_core_web_lg')
    spacy_de = spacy.load('de_core_news_sm')
    print('Finished loading spacy language models.')

    src = data.Field(tokenize=tokenize(spacy_en), lower=True, pad_token=CONSTANTS['pad'])
    tgt = data.Field(tokenize=tokenize(spacy_de),
                     lower=True,
                     init_token=CONSTANTS['start'],
                     pad_token=CONSTANTS['pad'],
                     eos_token=CONSTANTS['end'])

    print('Loading training vocabulary...')
    train_gen, val_gen, _ = datasets.Multi30k.splits(exts=('.en', '.de'),
                                                     fields=(('src', src), ('tgt', tgt)),
                                                     filter_pred=lambda x: len(vars(x)['src']) <= args.max_seq_length and len(vars(x)['tgt']) <= args.max_seq_length)
    
    src.build_vocab(train_gen.src, min_freq=args.min_word_freq)
    tgt.build_vocab(train_gen.tgt, min_freq=args.min_word_freq)
    print('Finished loading training vocabulary.')

    src_vocab_size = len(src.vocab.itos)
    tgt_vocab_size = len(tgt.vocab.itos)

    print('Loading dataset for translation...')
    dataset, _, _ = datasets.WMT14.splits(exts=('.en', '.de'),
                                          fields=(('src', src), ('tgt', tgt)),
                                          filter_pred=lambda x: len(vars(x)['src']) <= args.max_seq_length and len(vars(x)['tgt']) <= args.max_seq_length)
    print('Finished loading dataset for translation.')

    print('Instantiating model...')
    device = args.device
    model = Transformer(src_vocab_size, tgt_vocab_size, device, p_dropout=args.dropout)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    print('Model instantiated.')

    translate(model, dataset, src.vocab, tgt.vocab, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer Network')
    parser.add_argument('--no-cuda', action="store_true",
                        help='run on cpu')
    parser.add_argument('--max-seq-length', type=int, default=50,
                        help='maximum length of sentence to use (default: 50)')
    parser.add_argument('--min-word-freq', type=int, default=5,
                        help='minimum word frequency to be added to dictionary (default: 5)')
    parser.add_argument('--beam-size', type=int, default=5,
                        help='beam size to use in translation (default: 5)')
    parser.add_argument('--model', type=str, required=True,
                        help='path to model parameters')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    run(args)
