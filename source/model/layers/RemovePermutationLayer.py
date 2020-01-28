
import tensorflow as tf

class RemovePermutationLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(RemovePermutationLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs, mask=None):

        result = inputs[0:-1:self.get_permutation_count(), :]

        return result

    def get_permutation_count(self):
        return int(self.config["model"]["permutation-count"])

