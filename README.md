# Sequence-to-sequence by Tensorflow 2.0 Beta

This project is partly derived from [an original tutorial](https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention).

## Development Environment

- OS: Ubuntu 16.04
- Architecture: x86_64
- CUDA: 10.0, V10.0.130
- cuDNN: 7.6.0
- Python: 3.6.5
- tensorflow-gpu: 2.0.0b
- tqdm

## Run

1. Prepare the dataset, which is a English-Spanish parallel corpus.

```
$ python prepare_dataset.py ./dataset
```

2. Create a training configuration file.

```
$ python configure.py ./dataset ./config  # you will get ./config/config.json
```

3. Run the training script.

```
$ python train.py -c ./config/config.json -d <GPUID>
```
