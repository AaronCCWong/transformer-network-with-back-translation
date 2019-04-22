import argparse
import math
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


def train(model, epoch, train_iterator, optimizer, src_vocab, tgt_vocab, args, writer):
    model.train()

    losses = 0
    correct_words = 0
    total_words = 0

    for batch_idx, batch in enumerate(train_iterator):
        device = args.device
        src = batch.src.transpose(0, 1).to(device)
        tgt = batch.tgt.transpose(0, 1).to(device)
        src_mask = padding_mask(src, src_vocab)
        tgt_mask = padding_mask(tgt[:, :-1], src_vocab) & subsequent_mask(tgt[:, :-1]).to(device)

        out = model(src, tgt[:, :-1], src_mask, tgt_mask)
        optimizer.zero_grad()

        labels = tgt[:, 1:].contiguous().view(-1)
        loss, n_correct = cal_performance(out, labels, tgt_vocab)
        loss.backward()
        optimizer.step()

        losses += loss.item()
        total_words += tgt[:, 1:].ne(tgt_vocab.stoi[CONSTANTS['pad']]).sum().item()
        correct_words += n_correct

        if batch_idx % args.log_interval == 0:
            print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %'.format(
                  ppl=math.exp(losses / total_words), accu=100 * correct_words / total_words))
    writer.add_scalar('train_loss', losses / total_words, epoch)


def validate(model, epoch, val_iterator, src_vocab, tgt_vocab, args, writer):
    model.eval()

    losses = 0
    correct_words = 0
    total_words = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_iterator):
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

            print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %'.format(
                      ppl=math.exp(losses / total_words), accu=100 * correct_words / total_words))
    writer.add_scalar('val_loss', losses / total_words, epoch)


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
    device = args.device
    model = Transformer(src_vocab_size, tgt_vocab_size, device=device)
    model = model.to(device)
    print('Model instantiated!')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    print('Starting training...')
    for epoch in range(args.epochs):
        train(model, epoch + 1, train_iterator, optimizer, src.vocab, tgt.vocab, args, writer)
        validate(model, epoch + 1, val_iterator, src.vocab, tgt.vocab, args, writer)
        model_file = 'models/model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print('Saved model to ' + model_file)
    print('Finished training.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer Network')
    parser.add_argument('--epochs', type=int, default=10, metavar='E',
                        help='number of epochs to train for (default: 10)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='L',
                        help='number of batches to wait before logging training stats (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate of the decoder (default: 1e-4)')

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    run(args)
