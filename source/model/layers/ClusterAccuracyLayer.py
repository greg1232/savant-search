import tensorflow as tf

import numpy

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

class ClusterAccuracyLayer(tf.keras.layers.Layer):
    def __init__(self, config, window=512, update_step = 128, **kwargs):
        super(ClusterAccuracyLayer, self).__init__(trainable=False, **kwargs)
        self.config = config

        self.kmeans = KMeans(n_clusters=2, random_state=0)

        self.document_embeddings = numpy.zeros((window, self.get_embedding_size()), numpy.float32)
        self.classes = ["" for i in range(window)]

        self.window = window
        self.update_step = update_step
        self.step = 0

        self.accuracy = 0.0

    def call(self, inputs, training=None):
        loss, accuracy = self.add_inputs(inputs)

        self.add_metric(accuracy, name='cluster_accuracy', aggregation='mean')

        return loss

    def compute_output_shape(self, input_shape):
        return input_shape[2]

    def add_inputs(self, inputs):
        document_embeddings, classes, loss = inputs

        accuracy = self.compute_accuracy(document_embeddings, classes)

        loss = loss + tf.cast(self.accuracy, tf.float32)

        return loss, accuracy

    @tf.autograph.experimental.do_not_convert
    def compute_accuracy(self, document_embeddings, classes):
        def update_clusters_and_accuracy(document_embeddings_tensor, classes_tensor):

            batch_size = classes_tensor.numpy().shape[0]

            assert self.window % batch_size == 0, "Batch size does not evenly divide the window " + str(batch_size)

            self.classes[self.step:self.step + batch_size] = [class_name.decode('utf8') for class_name in classes_tensor.numpy()[:,0]]
            self.document_embeddings[self.step:self.step + batch_size, :] = document_embeddings_tensor.numpy()[:,:]

            self.step = (self.step + batch_size) % self.window
            if (self.step % self.update_step) != 0:
                return tf.convert_to_tensor(self.accuracy, dtype=tf.float64)

            classes = [class_name for class_name in self.classes if len(class_name) > 0]
            document_embeddings = self.document_embeddings[0:len(classes),:]

            class_map = {}
            for class_name in classes:
                if not class_name in class_map:
                    class_map[class_name] = len(class_map)

            class_numbers = numpy.array([class_map[class_name] for class_name in classes])

            predictions = self.kmeans.fit_predict(document_embeddings)

            one_hot_predictions = self.to_one_hot(predictions)
            one_hot_classes = self.to_one_hot(class_numbers)

            self.accuracy = max(accuracy_score(one_hot_predictions, one_hot_classes),
                accuracy_score(one_hot_predictions, self.invert(one_hot_classes)))

            return tf.convert_to_tensor(self.accuracy, dtype=tf.float64)

        return tf.py_function(update_clusters_and_accuracy, (document_embeddings, classes), tf.float64)

    def to_one_hot(self, indices):
        one_hot = numpy.zeros((indices.shape[0], 2))

        one_hot[numpy.arange(indices.size), indices] = 1

        return one_hot

    def invert(self, one_hot):
        return numpy.ones_like(one_hot) - one_hot

    def get_embedding_size(self):
        return int(self.config["model"]["layer-size"])



