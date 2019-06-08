"""Prepare dataset."""
from typing import Tuple, List
import argparse
import collections
import datetime
import os
import re

import tensorflow as tf
from sklearn.model_selection import train_test_split


def preprocess_text(text: str) -> str:
    # create a space between a word and the punctuation following it
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = text.strip()
    return text


def load_dataset(path: str) -> Tuple[List[str], List[str]]:
    src, tgt = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            src_, tgt_ = line.strip().split('\t')
            src.append(preprocess_text(src_))
            tgt.append(preprocess_text(tgt_))
    return src, tgt


def build_vocabulary(texts: List[str], max_vocab_size: int) -> List[str]:
    word_counts = collections.Counter(' '.join(texts).split(' '))
    return [word for word, _ in word_counts.most_common(max_vocab_size)]


def save_tokenzed_texts(output_filepath: str, texts: List[str]) -> None:
    with open(output_filepath, 'w') as f:
        for tokenized_text in texts:
            f.write(f'{tokenized_text}\n')


def save_tokenizer(output_filepath: str, vocab: List[str]) -> None:
    with open(output_filepath, 'w') as f:
        for word in vocab:
            f.write(f'{word}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('BUILD_PATH', help='path to resultant dataset')
    parser.add_argument('--max-vocab-size', default=50000, type=int, help='maximum vocabulary size')
    args = parser.parse_args()

    # download a dataset
    print(f'[{datetime.datetime.now()}] Downloading dataset...')
    path_to_zip = tf.keras.utils.get_file(
        fname='spa-eng.zip',
        origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
        extract=True
    )
    path_to_file = os.path.join(os.path.dirname(path_to_zip), 'spa-eng', 'spa.txt')

    # load the downloaded dataset
    print(f'[{datetime.datetime.now()}] Loading dataset...')
    src, tgt = load_dataset(path_to_file)

    src_train, src_test_, tgt_train, tgt_test_ = train_test_split(src, tgt, test_size=.2)
    src_valid, src_test, tgt_valid, tgt_test = train_test_split(src_test_, tgt_test_, test_size=.5)

    # build the vocabulary and tokenize texts
    print(f'[{datetime.datetime.now()}] Building vocabulary...')
    src_vocab = build_vocabulary(src_train, args.max_vocab_size)
    tgt_vocab = build_vocabulary(tgt_train, args.max_vocab_size)

    # save tokenized texts and tokenizers
    print(f'[{datetime.datetime.now()}] Saving...')
    os.makedirs(args.BUILD_PATH, exist_ok=True)
    save_tokenzed_texts(os.path.join(args.BUILD_PATH, 'src_train.txt'), src_train)
    save_tokenzed_texts(os.path.join(args.BUILD_PATH, 'tgt_train.txt'), tgt_train)
    save_tokenzed_texts(os.path.join(args.BUILD_PATH, 'src_valid.txt'), src_valid)
    save_tokenzed_texts(os.path.join(args.BUILD_PATH, 'tgt_valid.txt'), tgt_valid)
    save_tokenzed_texts(os.path.join(args.BUILD_PATH, 'src_test.txt'), src_test)
    save_tokenzed_texts(os.path.join(args.BUILD_PATH, 'tgt_test.txt'), tgt_test)
    save_tokenizer(os.path.join(args.BUILD_PATH, 'src_vocab.txt'), src_vocab)
    save_tokenizer(os.path.join(args.BUILD_PATH, 'tgt_vocab.txt'), tgt_vocab)


if __name__ == '__main__':
    main()
