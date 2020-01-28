
import tensorflow as tf

class AddPositionEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(AddPositionEncodingLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs, mask=None):

        embeddings, positions = inputs

        position_embeddings = self.positional_encoding(positions)
        result = tf.concat([embeddings, position_embeddings], -1)

        return position_embeddings + embeddings

    def compute_mask(self, inputs, mask=None):

        if mask is None:
            return None

        embeddings_mask = mask[0]

        return embeddings_mask

    def get_angles(self, pos, i):
        hidden_size = self.get_layer_size()

        dw = tf.constant(hidden_size, dtype=tf.float32)

        exp = 2.0 * (tf.cast(i, dtype=tf.float32) // 2.0)
        angle_rates = tf.exp(exp * -(tf.math.log(10000.0) / dw))

        return tf.cast(pos, dtype=tf.float32) * angle_rates

    def positional_encoding(self, positions):
        hidden_size = self.get_layer_size()

        batch_size = tf.shape(positions)[0]
        timesteps = tf.shape(positions)[1]

        angle_rads = self.get_angles(tf.expand_dims(positions,-1),
            tf.expand_dims(tf.expand_dims(tf.range(hidden_size), 0), 0))
        PE_cos = tf.cos(angle_rads[:, :, 0::2])
        PE_sin = tf.sin(angle_rads[:, :, 1::2])

        pos_encoding = tf.concat([PE_cos, PE_sin], axis=-1)

        pos_encoding = tf.reshape(pos_encoding, (batch_size, timesteps, self.get_layer_size()))

        return pos_encoding

    def get_layer_size(self):
        return int(self.config["model"]["layer-size"])

