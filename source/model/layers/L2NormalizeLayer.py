import tensorflow as tf

class L2NormalizeLayer(tf.keras.layers.Layer):
    def __init__(self, axis=0, **kwargs):
        super(L2NormalizeLayer, self).__init__(**kwargs)
        self.axis=axis

    def call(self, inputs, mask=None):
        return tf.math.l2_normalize(inputs, axis=self.axis)

