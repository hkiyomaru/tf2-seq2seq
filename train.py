from typing import Tuple
import argparse
import datetime
import json
import os

from tqdm import tqdm
import tensorflow as tf

PAD = 0
START = 1
END = 2
UNK = 3


class Encoder(tf.keras.Model):

    def __init__(self, vocab_size: int, embedding_dim: int, enc_units: int) -> None:
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True)

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.embedding(x)
        output, state = self.gru(x)
        return output, state


class BahdanauAttention(tf.keras.Model):

    def __init__(self, units: int) -> None:
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query: tf.Tensor, values: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x: tf.Tensor, hidden: tf.Tensor, enc_output: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights


def get_index_table_from_file(path: str) -> tf.lookup.StaticHashTable:
    table = tf.lookup.StaticHashTable(
        tf.lookup.TextFileInitializer(
            path,
            tf.string,
            tf.lookup.TextFileIndex.WHOLE_LINE,
            tf.int64,
            tf.lookup.TextFileIndex.LINE_NUMBER
        ),
        UNK - 4
    )
    return table


def get_dataset(src_path: str, table: tf.lookup.StaticHashTable) -> tf.data.Dataset:

    def to_ids(text):
        tokenized = tf.strings.split(tf.reshape(text, [1]), sep=' ')
        ids = table.lookup(tokenized.values) + 4  # 4 is for PAD, START, END, and UNK
        return ids

    def add_start_end_tokens(tokens):
        ids = tf.concat([[START], tf.cast(tokens, tf.int32), [END]], axis=0)
        return ids

    dataset = tf.data.TextLineDataset(src_path)
    dataset = dataset.map(to_ids)
    dataset = dataset.map(add_start_end_tokens)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='path to a configuration file')
    parser.add_argument('--device', '-d', help='path to preprocessed dataset')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    with open(args.config) as f:
        config = json.load(f)

    print(f'[{datetime.datetime.now()}] Loading vocabularies and making hash tables...')
    src_table = get_index_table_from_file(config['data']['src_vocab'])
    tgt_table = get_index_table_from_file(config['data']['tgt_vocab'])

    print(f'[{datetime.datetime.now()}] Loading pre-processed datasets...')
    src_train = get_dataset(config['data']['src_train'], src_table)
    tgt_train = get_dataset(config['data']['tgt_train'], tgt_table)
    train_dataset = tf.data.Dataset.zip((src_train, tgt_train))
    train_dataset = train_dataset.shuffle(config['other']['buffer_size'])
    train_dataset = train_dataset.padded_batch(
        config['other']['batch_size'],
        padded_shapes=([config['data']['src_max_length'] + 2], [config['data']['tgt_max_length'] + 2]),  # START + END
        padding_values=(PAD, PAD),
        drop_remainder=True
    )
    train_dataset = train_dataset.prefetch(2)

    # make a model
    print(f'[{datetime.datetime.now()}] Preparing models...')
    encoder = Encoder(
        config['data']['src_vocab_size'] + 4,  # 4 is for PAD, START, END, and UNK
        config['arch']['embedding_dim'],
        config['arch']['units']
    )
    decoder = Decoder(
        config['data']['tgt_vocab_size'] + 4,  # 4 is for PAD, START, END, and UNK
        config['arch']['embedding_dim'],
        config['arch']['units']
    )

    # define the optimizer and the loss function
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
        loss_ = loss_object(real, pred)
        mask = tf.math.logical_not(tf.math.equal(real, PAD))
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(src: tf.Tensor, tgt: tf.Tensor):
        batch_size, tgt_length = tgt.shape
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(src)
            dec_hidden = enc_hidden
            for t in range(tgt_length - 1):
                dec_input = tf.expand_dims(tgt[:, t], 1)  # using teacher forcing
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                loss += loss_function(tgt[:, t + 1], predictions)
        batch_loss = loss / tgt_length
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    print(f'[{datetime.datetime.now()}] Ready.')
    for epoch in range(config['other']['epoch']):
        total_loss = 0
        for batch, (src, tgt) in tqdm(enumerate(train_dataset)):
            batch_loss = train_step(src, tgt)
            total_loss += batch_loss
            if batch % 100 == 0:
                tqdm.write(f'Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy():.4f}')
        tqdm.write(f'Epoch {epoch + 1} Loss {total_loss:.4f}')


if __name__ == '__main__':
    main()
