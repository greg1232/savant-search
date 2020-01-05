
import os

import tensorflow as tf
import tensorflow_datasets as tfds

import random

import logging

logger = logging.getLogger(__name__)

class EncoderLayer:
    def __init__(self, config, training_dataset):
        self.config = config
        self.training_dataset = training_dataset

    def encode_inputs(self, inputs):
        with tf.device('/cpu:0'):
            if not self.does_vocab_file_exist():
                logger.debug("Building vocab from corpus...")
                self.encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                    self.training_dataset.get_raw_text_generator(), self.get_target_vocab_size(),
                    max_corpus_chars=self.get_maximum_corpus_size_for_vocab())
                logger.debug(" Finished...")
                self.encoder.save_to_file(self.get_vocab_path())

            self.encoder = tfds.features.text.SubwordTextEncoder.load_from_file(
                self.get_vocab_path())

            self.random = random.Random(2)

            def encode(encoded):

                encoded = [self.encoder.encode(str(x.numpy()[0])) for x in encoded]

                # add special tokens for document ends
                encoded   = [ [i + 3 for i in x]  for x in encoded ]
                labels    = [ x[1:] + [2]         for x in encoded ]
                positions = [ [1 + (i%127) for i in range(len(x))] for x in encoded ]

                zipped = [ list(zip(x, l, p)) for x, l, p in zip(encoded, labels, positions) ]

                # expand the batch size to add multiple permutations
                zipped = [ z for i in range(self.get_permutation_count()) for z in zipped ]

                # shuffle
                for x in zipped:
                    self.random.shuffle(x)

                    # add special tokens for embeddings
                    x.append((1, 1, 1 + (len(x) % 127)))

                encoded   = [ [e for e, l, p in x] for x in zipped]
                labels    = [ [l for e, l, p in x] for x in zipped]
                positions = [ [p for e, l, p in x] for x in zipped]

                # pad
                max_length = max([len(x) for x in encoded])

                encoded   = [x + [0 for i in range(max_length - len(x))] for x in encoded]
                labels    = [x + [0 for i in range(max_length - len(x))] for x in labels]
                positions = [x + [0 for i in range(max_length - len(x))] for x in positions]

                # convert to tensors
                encoded = tf.convert_to_tensor(list(zip(encoded, labels, positions)), dtype=tf.int64)

                return encoded

            result = tf.keras.layers.Lambda(
                lambda inputs: tf.py_function(encode, [inputs], tf.int64))(inputs)

            result.set_shape((None, 3, None))

            encoded   = result[:, 0, :]
            labels    = result[:, 1, :]
            positions = result[:, 2, :]

            encoded.set_shape((None, None))
            labels.set_shape((None, None))
            positions.set_shape((None, None))

            return encoded, labels, positions

    def get_vocab_path(self):
        return os.path.join(self.config['model']['directory'], 'vocab')

    def get_target_vocab_size(self):
        return int(self.config['model']['vocab-size'])

    def get_vocab_size(self):
        return self.encoder.vocab_size

    def does_vocab_file_exist(self):
        return os.path.exists(self.get_vocab_path() + ".subwords")

    def get_permutation_count(self):
        return int(self.config['model']['permutation-count'])

    def get_maximum_corpus_size_for_vocab(self):
        return int(self.config['model']['maximum-corpus-size-for-vocab'])




