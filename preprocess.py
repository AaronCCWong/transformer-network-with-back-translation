import argparse
import csv
import pandas as pd
from sklearn.model_selection import train_test_split


def process_data(filepath):
    f = open(filepath, 'r')
    raw_data = {'src': [line for line in f]}
    f.close()

    df = pd.DataFrame(raw_data, columns=['src'])
    df['de_len'] = df['src'].str.count(' ')
    df = df.query('de_len <= 50')
    train, val = train_test_split(df, test_size=0)
    train.to_csv('data/train.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess news data')
    parser.add_argument('--path', type=str, default='data/news.2007.de.shuffled',
                        help='path to news monolingual data file')

    args = parser.parse_args()
    print('Running with these options:', args)
    process_data(args.path)
