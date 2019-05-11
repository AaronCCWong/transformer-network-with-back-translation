import csv
import pandas as pd
from sklearn.model_selection import train_test_split

from transformer.utils import LANGUAGES, get_tokenizer, tokenize


def process_data(srcpath, tgtpath):
    f_src = open(srcpath, 'r')
    f_tgt = open(tgtpath, 'r')
    raw_data = {'src': [line for line in f_src], 'tgt': [line for line in f_tgt]}
    f_src.close()
    f_tgt.close()

    df = pd.DataFrame(raw_data, columns=['src', 'tgt'])
    df['en_len'] = df['src'].str.count(' ')
    df['de_len'] = df['tgt'].str.count(' ')
    df = df.query('de_len <= 50 & en_len <= 50')
    train, val = train_test_split(df, test_size=0)
    train.to_csv('data/train.csv', index=False)


if __name__ == "__main__":
    process_data('data/src.txt', 'data/tgt.txt')
