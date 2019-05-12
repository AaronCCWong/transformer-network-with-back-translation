import argparse
import torch
from torchtext import data, datasets

from translator.translator import Translator
from transformer.utils import (CONSTANTS, get_tokenizer, build_file_extension, build_dataset)


def run(args):
    with torch.no_grad():
        src, tgt, _, _ = build_dataset(args)

        _, _, test_gen = datasets.IWSLT.splits(exts=(build_file_extension(args.src_language), build_file_extension(args.tgt_language)),
                                               fields=(('src', src), ('tgt', tgt)),
                                               filter_pred=lambda x: len(vars(x)['src']) <= args.max_seq_length and len(vars(x)['tgt']) <= args.max_seq_length)

        _, _, test_iterator = data.Iterator.splits((_, _, test_gen),
                                                    sort_key=lambda x: len(x.src),
                                                    batch_sizes=(args.batch_size, args.batch_size, args.batch_size))

        src_vocab_size = len(src.vocab.itos)
        tgt_vocab_size = len(tgt.vocab.itos)

        translator = Translator(src.vocab, tgt.vocab, src_vocab_size, tgt_vocab_size, args)

        with open('data/tgt.txt', 'w') as tgt_f:
            with open('data/src.txt', 'w') as src_f:
                for batch_idx, batch in enumerate(test_iterator):
                    tgt_seqs = batch.src.transpose(0, 1)
                    for idx_seqs in tgt_seqs:
                        sentence_idxs = [idx for idx in idx_seqs if idx not in (
                            src.vocab.stoi[CONSTANTS['pad']], src.vocab.stoi[CONSTANTS['start']], src.vocab.stoi[CONSTANTS['end']])]
                        line = ' '.join([src.vocab.itos[idx] for idx in sentence_idxs])
                        tgt_f.write(line + '\n')

                    all_hyp, all_scores = translator.translate_batch(batch.src.transpose(0, 1))
                    for idx_seqs in all_hyp:
                        for idx_seq in idx_seqs:
                            pred_line = ' '.join([tgt.vocab.itos[idx] for idx in idx_seq[:-1]])
                            src_f.write(pred_line + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer Network')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument('--no-cuda', action="store_true",
                        help='run on cpu')
    parser.add_argument('--max-seq-length', type=int, default=50,
                        help='maximum length of sentence to use (default: 50)')
    parser.add_argument('--min-word-freq', type=int, default=5,
                        help='minimum word frequency to be added to dictionary (default: 5)')
    parser.add_argument('--beam-size', type=int, default=5,
                        help='beam size to use in translation (default: 5)')
    parser.add_argument('--src-language', type=str, default='en',
                        help='the source language to translate from (default: en)')
    parser.add_argument('--tgt-language', type=str, default='de',
                        help='the source language to translate from (default: de)')
    parser.add_argument('--model', type=str, required=True,
                        help='path to model parameters')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size to use (default: 64)')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    print('Running with these options:', args)
    run(args)
