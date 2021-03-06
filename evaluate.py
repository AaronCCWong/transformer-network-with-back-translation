import argparse
import math
import spacy
import torch
import torchtext
from itertools import chain
from nltk.translate.bleu_score import corpus_bleu
from tensorboardX import SummaryWriter
from torchtext import data, datasets
from tqdm import tqdm

from transformer.transformer import Transformer
from transformer.utils import (CONSTANTS, cal_performance, padding_mask, subsequent_mask,
                               get_tokenizer, build_file_extension, build_dataset)


def test(model, test_iterator, src_vocab, tgt_vocab, args, writer):
    model.eval()

    losses = 0
    correct_words = 0
    total_words = 0

    references = []
    hypotheses = []
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_iterator), total=len(test_iterator)):
            device = args.device
            src = batch.src.transpose(0, 1).to(device)
            tgt = batch.tgt.transpose(0, 1).to(device)
            src_mask = padding_mask(src, src_vocab)
            tgt_mask = padding_mask(tgt[:, :-1], src_vocab) & subsequent_mask(tgt[:, :-1]).to(device)

            out = model(src, tgt[:, :-1], src_mask, tgt_mask)
            labels = tgt[:, 1:].contiguous().view(-1)
            loss, n_correct = cal_performance(out, labels, tgt_vocab)

            losses += loss.item()
            total_words += tgt[:, 1:].ne(tgt_vocab.stoi[CONSTANTS['pad']]).sum().item()
            correct_words += n_correct

            # Convert target sentence into words
            for idxs in tgt.tolist():
                references.append([[idx for idx in idxs
                                    if idx != tgt_vocab.stoi[CONSTANTS['start']] and idx != tgt_vocab.stoi[CONSTANTS['pad']]]])

            # Convert prediction into a sentence
            word_idxs = torch.max(out, dim=-1)[1]
            for idxs in word_idxs.tolist():
                hypotheses.append([idx for idx in idxs
                                   if idx != tgt_vocab.stoi[CONSTANTS['start']] and idx != tgt_vocab.stoi[CONSTANTS['pad']]])

    bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    print('(Test) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f}%, BLEU-1: {bleu1:3.3f}, BLEU-2: {bleu2:3.3f}'.format(
          ppl=math.exp(losses / total_words), accu=100 * correct_words / total_words, bleu1=bleu_1, bleu2=bleu_2))


def run(args):
    writer = SummaryWriter()
    src, tgt, _, _ = build_dataset(args)

    print('Loading test data split.')
    _, _, test_gen = datasets.Multi30k.splits(exts=(build_file_extension(args.src_language), build_file_extension(args.tgt_language)),
                                              fields=(('src', src), ('tgt', tgt)),
                                              filter_pred=lambda x: len(vars(x)['src']) <= args.max_seq_length and len(vars(x)['tgt']) <= args.max_seq_length)
    print('Finished loading test data split.')

    src_vocab_size = len(src.vocab.itos)
    tgt_vocab_size = len(tgt.vocab.itos)

    _, _, test_iterator = data.Iterator.splits((_, _, test_gen),
                                                sort_key=lambda x: len(x.src),
                                                batch_sizes=(args.batch_size, args.batch_size, args.batch_size))

    print('Instantiating model...')
    device = args.device
    model = Transformer(src_vocab_size, tgt_vocab_size, device, p_dropout=args.dropout)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model))
    print('Model instantiated!')

    print('Starting testing...')
    test(model, test_iterator, src.vocab, tgt.vocab, args, writer)
    print('Finished testing.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer Network')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='probability of dropout (default: 0.1)')
    parser.add_argument('--max-seq-length', type=int, default=50,
                        help='maximum length of sentence to use (default: 50)')
    parser.add_argument('--min-word-freq', type=int, default=5,
                        help='minimum word frequency to be added to dictionary (default: 5)')
    parser.add_argument('--src-language', type=str, default='en',
                        help='the source language to translate from (default: en)')
    parser.add_argument('--tgt-language', type=str, default='de',
                        help='the source language to translate from (default: de)')
    parser.add_argument('--model', type=str, required=True,
                        help='path to model parameters')
    parser.add_argument('--no-cuda', action="store_true",
                        help='run on cpu')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    print('Running with these options:', args)
    run(args)
