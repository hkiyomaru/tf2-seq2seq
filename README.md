# Sequence-to-sequence using Tensorflow 2

This project is partly derived from [the official tutorial](https://www.tensorflow.org/tutorials/text/nmt_with_attention), an example which trains a sequence-to-sequence (seq2seq) model for Spanish to English translation.

## Installation

Please follow [the official guide](https://www.tensorflow.org/install) to install Tensorflow.
In addition, you need to install [scikit-learn](https://scikit-learn.org/stable/install.html).

## Getting Started

### Preparing a Dataset

`prepare_dataset.py` downloads a parallel corpus with some preprocessing.
The resultant files will consume about 8.1MB.

```
$ python prepare_dataset.py <directory-to-save-dataset>
```

### Training a Seq2seq Model

`train.py` starts training a seq2seq model on the dataset.

```
$ python train.py --dataset <directory-to-save-dataset> --device 0
```
