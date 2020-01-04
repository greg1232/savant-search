
import tensorflow as tf
import tensorflow_addons as tfa

class ContrastivePredictiveCodingLossLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(ContrastivePredictiveCodingLossLayer, self).__init__(**kwargs)
        self.config = config

        self.triplet_loss = tfa.losses.TripletSemiHardLoss(self.get_margin())
        self.cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, inputs, mask=None):
        self.labels, self.output_embeddings, self.output_probabilities = inputs
        self.labels_mask, self.output_embeddings_mask, self.output_probabilities_mask = mask

        contrastive_loss = self.triplet_loss.call(
            self.get_batch_ids(), self.get_output_embeddings())

        self.add_metric(contrastive_loss, name='contrastive_loss', aggregation='mean')

        predictive_loss = self.cross_entropy_loss.call(self.get_true_classes(),
                                                       self.get_output_probabilities())

        self.add_metric(predictive_loss, name='predictive_loss', aggregation='mean')

        contrastive_loss *= self.get_contrastive_loss_scale()
        predictive_loss *= self.get_predictive_loss_scale()

        complete_loss = (contrastive_loss + tf.reduce_mean(predictive_loss))

        return complete_loss

    def get_batch_ids(self):

        batch_size = tf.shape(self.output_embeddings)[0] // self.get_permutation_count()
        timesteps = tf.shape(self.output_embeddings)[1]

        mask = self.labels_mask

        batch_ids = tf.reshape(tf.cast(tf.range(batch_size, dtype=tf.int32), tf.int64), (batch_size, 1))

        ids = tf.reshape(tf.zeros((batch_size, self.get_permutation_count()), dtype=tf.int64) + batch_ids, (-1,))

        return ids

    def get_output_embeddings(self):
        batch_size = tf.shape(self.labels_mask)[0]

        lengths = tf.reshape(tf.reduce_sum(tf.cast(self.labels_mask, tf.int64), axis=1) - 1, (batch_size, 1))

        embeddings = tf.gather(self.output_embeddings, lengths, batch_dims=1, axis=1)

        embeddings = tf.reshape(embeddings, (-1, self.output_embeddings.shape[2]))

        return embeddings

    def get_true_classes(self):
        mask = self.labels_mask

        return tf.boolean_mask(
            tf.reshape(self.labels, (-1, 1)),
            tf.reshape(mask, (-1,)))

    def get_output_probabilities(self):
        mask = self.labels_mask

        return tf.boolean_mask(
            tf.reshape(self.output_probabilities, (-1, self.output_probabilities.shape[2])),
            tf.reshape(mask, (-1,)))


    def get_margin(self):
        return float(self.config["model"]["triplet-margin"])

    def get_contrastive_loss_scale(self):
        return float(self.config["model"]["contrastive-scale"])

    def get_predictive_loss_scale(self):
        return float(self.config["model"]["predictive-scale"])

    def get_permutation_count(self):
        return int(self.config["model"]["permutation-count"])


