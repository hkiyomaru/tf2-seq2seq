"""Prepare dataset."""
import argparse
import collections
import datetime
import typing

import os
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split


def preprocess_sentence(text: str) -> str:
    # create a space between a word and the punctuation following it
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = text.strip()
    return text


def load_dataset(path: str) -> typing.Tuple[typing.List[str], typing.List[str]]:
    src, tgt = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            src_, tgt_ = line.strip().split('\t')
            src.append(preprocess_sentence(src_))
            tgt.append(preprocess_sentence(tgt_))
    return src, tgt


def create_vocabulary(texts: typing.List[str], max_vocab_size: int) -> typing.List[str]:
    word_counts = collections.Counter(' '.join(texts).split(' '))
    return [word for word, _ in word_counts.most_common(max_vocab_size)]


def save_tokenized_sentences(output_filepath: str, texts: typing.List[str]) -> None:
    with open(output_filepath, 'w') as f:
        for tokenized_text in texts:
            f.write(f'{tokenized_text}\n')


def save_vocabulary(output_filepath: str, vocab: typing.List[str]) -> None:
    with open(output_filepath, 'w') as f:
        for word in vocab:
            f.write(f'{word}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('BUILD_PATH', help='path to save dataset')
    parser.add_argument('--max-vocab-size', default=50000, type=int, help='maximum vocabulary size')
    args = parser.parse_args()

    # download a dataset
    print(f'[{datetime.datetime.now()}] Downloading a dataset...')
    path_to_zip = tf.keras.utils.get_file(
        fname='spa-eng.zip',
        origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
        extract=True
    )
    path_to_file = os.path.join(os.path.dirname(path_to_zip), 'spa-eng', 'spa.txt')

    # load the downloaded dataset
    print(f'[{datetime.datetime.now()}] Loading the dataset...')
    src, tgt = load_dataset(path_to_file)

    # split the dataset into train, validation, and test
    print(f'[{datetime.datetime.now()}] Splitting the dataset into training, validation, and test sets...')
    src_train, src_valid_test, tgt_train, tgt_valid_test = train_test_split(src, tgt, test_size=.2)
    src_valid, src_test, tgt_valid, tgt_test = train_test_split(src_valid_test, tgt_valid_test, test_size=.5)

    # create the vocabulary
    print(f'[{datetime.datetime.now()}] Creating the vocabulary...')
    src_vocab = create_vocabulary(src_train, args.max_vocab_size)
    tgt_vocab = create_vocabulary(tgt_train, args.max_vocab_size)

    # save tokenized texts
    print(f'[{datetime.datetime.now()}] Saving the preprocessed data...')
    os.makedirs(args.BUILD_PATH, exist_ok=True)
    save_tokenized_sentences(os.path.join(args.BUILD_PATH, 'src_train.txt'), src_train)
    save_tokenized_sentences(os.path.join(args.BUILD_PATH, 'tgt_train.txt'), tgt_train)
    save_tokenized_sentences(os.path.join(args.BUILD_PATH, 'src_valid.txt'), src_valid)
    save_tokenized_sentences(os.path.join(args.BUILD_PATH, 'tgt_valid.txt'), tgt_valid)
    save_tokenized_sentences(os.path.join(args.BUILD_PATH, 'src_test.txt'), src_test)
    save_tokenized_sentences(os.path.join(args.BUILD_PATH, 'tgt_test.txt'), tgt_test)
    save_vocabulary(os.path.join(args.BUILD_PATH, 'src_vocab.txt'), src_vocab)
    save_vocabulary(os.path.join(args.BUILD_PATH, 'tgt_vocab.txt'), tgt_vocab)


if __name__ == '__main__':
    main()
