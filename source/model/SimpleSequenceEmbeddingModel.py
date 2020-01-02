
import os

import tensorflow as tf

from model.layers.EncoderLayer import EncoderLayer
from model.layers.ContrastivePredictiveCodingLossLayer import ContrastivePredictiveCodingLossLayer
from model.layers.DummyLoss import DummyLoss

import logging

logger = logging.getLogger(__name__)

class SimpleSequenceEmbeddingModel:
    def __init__(self, config, training_dataset, validation_dataset):
        self.config = config
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

        self.create_or_load_model()

    def train(self):

        with tf.device('/cpu:0'):
            self.model.fit(x=self.training_dataset.get_tensorflow_dataset(),
                validation_data=self.validation_dataset.get_tensorflow_dataset(),
                epochs=self.get_epochs(),
                callbacks=self.get_callbacks())

        self.checkpoint()

    def get_callbacks(self):
        return [
            # Interrupt training if `val_loss` stops improving for over 2 epochs
            #tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            tf.keras.callbacks.ModelCheckpoint(self.get_best_model_directory(), mode='max',
                save_best_only=True, verbose=1, save_weights_only=True, monitor='val_cpc'),
            # Write TensorBoard logs to `./logs` directory
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.config['model']['directory'], 'logs'),
                update_freq=500)
        ]

    def predict_on_batch(self, x):
        return self.model.predict_on_batch(x)

    def checkpoint(self):
        self.model.save_weights(self.get_checkpoint_model_directory())

    def create_or_load_model(self):

        logger.debug("Loading or creating model from directory: " +
            self.config['model']['directory'])

        self.create_model()

        if self.does_model_exist():
            self.load_model()

    def does_model_exist(self):
        if os.path.exists(self.get_best_model_directory()):
            return True

        if os.path.exists(self.get_checkpoint_model_directory()):
            return True

        return False

    def load_model(self):
        path = self.get_checkpoint_model_directory()

        if os.path.exists(self.get_best_model_directory()):
            path = self.get_best_model_directory()

        self.model.load_weights(path, by_name=True)
        logger.debug("Loading model from : " + path)

    def create_model(self):
        inputs = tf.keras.Input(shape=(None,), dtype=tf.string)

        input_embeddings, labels = self.compute_embeddings(inputs)

        hidden = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.get_layer_size()))(input_embeddings)
        output_embeddings = tf.keras.layers.Conv1D(self.get_layer_size(), 3, padding='causal')(hidden)
        output_probabilities = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.get_input_vocab_size()))(output_embeddings)

        loss = ContrastivePredictiveCodingLossLayer(self.config)([labels, output_embeddings, output_probabilities])

        model = tf.keras.Model(inputs=inputs, outputs=loss)

        model.compile(optimizer=tf.keras.optimizers.Adam(self.get_learning_rate()),
              loss=DummyLoss(),
              metrics=[])

        print(model.summary())

        self.model = model

    def compute_embeddings(self, inputs):

        encoded_inputs, labels = self.encode_inputs(inputs)
        labels = tf.keras.layers.Reshape((-1, 1))(labels)
        labels = tf.keras.layers.Masking(mask_value=0)(labels)

        input_embeddings = tf.keras.layers.Embedding(self.get_input_vocab_size(),
            self.get_layer_size(), mask_zero=True)(encoded_inputs)
        hidden = tf.keras.layers.Reshape((-1, self.get_layer_size()))(input_embeddings)
        hidden._keras_mask = input_embeddings._keras_mask

        return hidden, labels

    def encode_inputs(self, inputs):
        self.encoder_layer = EncoderLayer(self.config, self.training_dataset)

        return self.encoder_layer.encode_inputs(inputs)

    def get_vocab_size(self):
        return self.encoder_layer.get_vocab_size()

    def get_input_vocab_size(self):
        # one for the zero masked value
        # one for the special embedding token
        return self.encoder_layer.get_vocab_size() + 3

    def get_layer_size(self):
        return int(self.config["model"]["layer-size"])

    def get_epochs(self):
        return int(self.config["model"]["epochs"])

    def get_learning_rate(self):
        return float(self.config["model"]["learning-rate"])

    def get_best_model_directory(self):
        return os.path.join(self.config['model']['directory'], 'best.h5')

    def get_checkpoint_model_directory(self):
        return os.path.join(self.config['model']['directory'], 'checkpoint.h5')

