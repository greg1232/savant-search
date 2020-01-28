

import tensorflow as tf

class ExtractEmbeddingsLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(ExtractEmbeddingsLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs, mask=None):
        embeddings, labels = inputs
        embeddings_mask, labels_mask = mask

        batch_size = tf.shape(labels_mask)[0]

        lengths = tf.reshape(tf.reduce_sum(tf.cast(labels_mask, tf.int64), axis=1) - 1, (batch_size, 1))

        embeddings = tf.gather(embeddings, lengths, batch_dims=1, axis=1)

        embeddings = tf.reshape(embeddings, (batch_size, embeddings.shape[-1]))

        return embeddings

