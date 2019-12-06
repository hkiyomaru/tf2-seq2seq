# Sequence-to-sequence using Tensorflow 2

This project is partly derived from [an official tutorial](https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention).

## Installation

Please follow [the official guide](https://www.tensorflow.org/install) to install Tensorflow.
In addition, you need to install scikit-learn.

## Getting Started

### Dataset preparation

`prepare_dataset.py` downloads a parallel corpus with some preprocessing.
The resultant files will consume about 8.1MB.

```
$ python prepare_dataset.py ./dataset
```

### Training a seq2seq model

`train.py` starts training a seq2seq model.

```
$ python train.py --dataset ./dataset --device 0
```
