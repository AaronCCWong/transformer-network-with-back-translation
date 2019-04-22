import argparse
import spacy
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchtext
from tensorboardX import SummaryWriter
from torchtext import data, datasets

from transformer import Transformer
from utils import AverageMeter, CONSTANTS, padding_mask, subsequent_mask, tokenize


MAX_SEQ_LEN = 50
MIN_WORD_FREQ = 2


def train(model, epoch, train_iterator, optimizer, src_vocab, tgt_vocab, log_interval, writer):
    model.train()

    losses = AverageMeter()
    for batch_idx, batch in enumerate(train_iterator):
        src = batch.src.transpose(0, 1)
        tgt = batch.tgt.transpose(0, 1)
        src_mask = padding_mask(src, src_vocab)
        tgt_mask = padding_mask(tgt[:, :-1], src_vocab) & subsequent_mask(tgt[:, :-1])

        out = model(src, tgt[:, :-1], src_mask, tgt_mask)
        optimizer.zero_grad()

        labels = tgt[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(out.view(-1, out.size(-1)), labels,
                                ignore_index=tgt_vocab.stoi[CONSTANTS['pad']])
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), src.size(0))
        if batch_idx % log_interval == 0:
            print('Train Batch: [{0}/{1}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        batch_idx, len(train_iterator), loss=losses))
    writer.add_scalar('train_loss', losses.avg, epoch)


def validate(model, epoch, val_iterator, src_vocab, tgt_vocab, log_interval, writer):
    model.eval()

    losses = AverageMeter()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_iterator):
            src = batch.src.transpose(0, 1)
            tgt = batch.tgt.transpose(0, 1)
            src_mask = padding_mask(src, src_vocab)
            tgt_mask = padding_mask(tgt[:, :-1], src_vocab) & subsequent_mask(tgt[:, :-1])

            out = model(src, tgt[:, :-1], src_mask, tgt_mask)
            labels = tgt[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(out.view(-1, out.size(-1)), labels,
                                   ignore_index=tgt_vocab.stoi[CONSTANTS['pad']])

            losses.update(loss.item(), src.size(0))
            if batch_idx % log_interval == 0:
                print('Validation Batch: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          batch_idx, len(val_iterator), loss=losses))
    writer.add_scalar('val_loss', losses.avg, epoch)


def run(args):
    writer = SummaryWriter()

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

    print('Loading data splits...')
    train_gen, val_gen, test_gen = datasets.Multi30k.splits(exts=('.en', '.de'),
                                                fields=(('src', src), ('tgt', tgt)),
                                                filter_pred=lambda x: len(vars(x)['src']) <= MAX_SEQ_LEN and len(vars(x)['tgt']) <= MAX_SEQ_LEN)
    print('Finished loading data splits.')

    print('Building vocabulary...')
    src.build_vocab(train_gen.src, min_freq=MIN_WORD_FREQ)
    tgt.build_vocab(train_gen.tgt, min_freq=MIN_WORD_FREQ)
    print('Finished building vocabulary.')

    src_vocab_size = len(src.vocab.itos)
    tgt_vocab_size = len(tgt.vocab.itos)

    train_iterator, val_iterator, test_iterator = data.Iterator.splits((train_gen, val_gen, test_gen),
                                                                        sort_key=lambda x: len(x.src),
                                                                        batch_sizes=(32, 256, 256))

    print('Intstantiating model...')
    model = Transformer(src_vocab_size, tgt_vocab_size)
    print('Model instantiated!')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    print('Starting training...')
    for epoch in range(args.epochs):
        train(model, epoch + 1, train_iterator, optimizer, src.vocab, tgt.vocab, args.log_interval, writer)
        validate(model, epoch + 1, val_iterator, src.vocab, tgt.vocab, args.log_interval, writer)
    print('Finished training.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer Network')
    parser.add_argument('--epochs', type=int, default=10, metavar='E',
                        help='number of epochs to train for (default: 10)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='L',
                        help='number of batches to wait before logging training stats (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate of the decoder (default: 1e-4)')

    run(parser.parse_args())
