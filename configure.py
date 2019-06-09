"""Prepare dataset."""
import argparse
import datetime
import json
import os


def count_lines(path: str) -> int:
    with open(path) as f:
        return len([_ for _ in f])


def count_max_length(path: str) -> int:
    with open(path) as f:
        return max(len(line.strip().split(' ')) for line in f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('DATASET', help='path to preprocessed datasets')
    parser.add_argument('BUILD_PATH', help='path to the resultant configuration directory')
    parser.add_argument('--epoch', '-e', default=10, type=int, help='epoch number')
    parser.add_argument('--batchsize', '-b', default=64, type=int, help='batch size')
    parser.add_argument('--embedding-dim', default=256, type=int, help='number of dimension in an embedding space')
    parser.add_argument('--units', default=1024, type=int, help='number of dimension in hidden representations')
    parser.add_argument('--buffersize', default=4096, type=int, help='buffer size for shuffling dataset')
    args = parser.parse_args()

    basedir = os.path.abspath(args.DATASET)
    config = {
        'data': {
            'basedir': basedir,
            'src_train': os.path.join(basedir, 'src_train.txt'),
            'tgt_train': os.path.join(basedir, 'tgt_train.txt'),
            'src_valid': os.path.join(basedir, 'src_valid.txt'),
            'tgt_valid': os.path.join(basedir, 'tgt_valid.txt'),
            'src_test': os.path.join(basedir, 'src_test.txt'),
            'tgt_test': os.path.join(basedir, 'tgt_test.txt'),
            'src_vocab': os.path.join(basedir, 'src_vocab.txt'),
            'tgt_vocab': os.path.join(basedir, 'tgt_vocab.txt'),
            'src_max_length': max([
                count_max_length(os.path.join(basedir, 'src_train.txt')),
                count_max_length(os.path.join(basedir, 'src_valid.txt')),
                count_max_length(os.path.join(basedir, 'src_test.txt'))]),
            'tgt_max_length': max([
                count_max_length(os.path.join(basedir, 'tgt_train.txt')),
                count_max_length(os.path.join(basedir, 'tgt_valid.txt')),
                count_max_length(os.path.join(basedir, 'tgt_test.txt'))]),
            'src_vocab_size': count_lines(os.path.join(basedir, 'src_vocab.txt')),
            'tgt_vocab_size': count_lines(os.path.join(basedir, 'tgt_vocab.txt')),
        },
        'arch': {
            'embedding_dim': args.embedding_dim,
            'units': args.units,

        },
        'other': {
            'epoch': args.epoch,
            'batch_size': args.batchsize,
            'buffer_size': args.buffersize
        }
    }

    print(f'[{datetime.datetime.now()}] Writing a configuration file...')
    os.makedirs(args.BUILD_PATH, exist_ok=True)
    with open(os.path.join(args.BUILD_PATH, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)


if __name__ == '__main__':
    main()
