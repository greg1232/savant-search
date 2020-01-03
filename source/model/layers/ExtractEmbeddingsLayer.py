

import tensorflow as tf
import tensorflow_addons as tfa

class ExtractEmbeddingsLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(ExtractEmbeddingsLayer, self).__init__(**kwargs)
        self.config = config

    def call(inputs, mask=None):
        embeddings, labels = inputs
        embeddings_mask, labels_mask = mask

        batch_size = tf.shape(labels_mask)[0]

        lengths = tf.reshape(tf.reduce_sum(tf.cast(self.labels_mask, tf.int64), axis=1) - 1, (batch_size, 1))

        embeddings = tf.gather(embeddings, lengths, batch_dims=1, axis=1)

        return embeddings[0:-1:self.get_permutation_count(),:]

