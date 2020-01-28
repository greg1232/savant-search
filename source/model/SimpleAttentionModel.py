
import os

import tensorflow as tf

from model.layers.EncoderLayer import EncoderLayer
from model.layers.L2NormalizeLayer import L2NormalizeLayer
from model.layers.ContrastivePredictiveCodingLossLayer import ContrastivePredictiveCodingLossLayer
from model.layers.ExtractEmbeddingsLayer import ExtractEmbeddingsLayer
from model.layers.RemovePermutationLayer import RemovePermutationLayer
from model.layers.AddPositionEncodingLayer import AddPositionEncodingLayer
from model.layers.DummyLoss import DummyLoss
from model.layers.ClusterAccuracyLayer import ClusterAccuracyLayer

import logging

logger = logging.getLogger(__name__)

class SimpleAttentionModel:
    def __init__(self, config, training_dataset, validation_dataset):
        self.config = config
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

        self.create_or_load_model()

    def train(self):

        with tf.device('/cpu:0'):
            self.training_model.fit(x=self.training_dataset.get_tensorflow_dataset(),
                validation_data=self.validation_dataset.get_tensorflow_dataset(),
                epochs=self.get_epochs(),
                callbacks=self.get_callbacks())

        self.checkpoint()

    def get_callbacks(self):
        return [
            # Interrupt training if `val_loss` stops improving for over 2 epochs
            #tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            tf.keras.callbacks.ModelCheckpoint(self.get_best_model_directory(), mode='min',
                save_best_only=True, verbose=1, save_weights_only=True, monitor='loss',
                save_freq=100000),
            # Write TensorBoard logs to `./logs` directory
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.config['model']['directory'], 'logs'),
                profile_batch=self.get_profile_batch(),
                update_freq=100)
        ]

    def predict_on_batch(self, x):
        return self.embedding_model.predict_on_batch(x)

    def checkpoint(self):
        self.training_model.save_weights(self.get_checkpoint_model_directory())

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

        self.training_model.load_weights(path, by_name=True)
        logger.debug("Loading model from : " + path)

    def create_model(self):
        inputs = tf.keras.Input(shape=(None,), dtype=tf.string)
        classes = tf.keras.Input(shape=(None,), dtype=tf.string)

        input_embeddings, labels = self.compute_embeddings(inputs)

        hidden = input_embeddings

        #for layer in range(self.get_layer_count()):
        #    hidden = self.add_attention_layer(hidden)

        output_embeddings = L2NormalizeLayer(axis=2)(hidden)

        output_probabilities = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.get_input_vocab_size()))(output_embeddings)

        document_embeddings = self.get_document_embeddings(output_embeddings, labels)

        loss = ContrastivePredictiveCodingLossLayer(self.config)(
            [labels, document_embeddings, output_probabilities])

        document_embeddings = RemovePermutationLayer(self.config)(document_embeddings)

        loss = ClusterAccuracyLayer(self.config)([document_embeddings, classes, loss])

        self.training_model = tf.keras.Model(inputs=[inputs, classes], outputs=loss)

        self.training_model.compile(optimizer=tf.keras.optimizers.Adam(self.get_learning_rate()),
              loss=DummyLoss(),
              metrics=[])

        print(self.training_model.summary())

        self.embedding_model = tf.keras.Model(inputs=inputs, outputs=document_embeddings)

    def add_attention_layer(self, hidden):

        query = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.get_layer_size(), activation='relu'))(hidden)
        value = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.get_layer_size(), activation='relu'))(hidden)
        updated = tf.keras.layers.Attention(use_scale=True, causal=True, dropout=self.get_dropout())([query, value])

        updated = tf.keras.layers.Add()([updated, hidden])
        updated = tf.keras.layers.LayerNormalization()(updated)
        attention_result = tf.keras.layers.Dropout(self.get_dropout())(updated)

        updated = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.get_layer_size(), activation='relu'))(attention_result)
        updated = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.get_layer_size()))(updated)

        updated = tf.keras.layers.Add()([attention_result, updated])
        updated = tf.keras.layers.LayerNormalization()(updated)
        updated = tf.keras.layers.Dropout(self.get_dropout())(updated)

        return updated

    def get_document_embeddings(self, output_embeddings, labels):
        output_embeddings = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.get_layer_size()))(output_embeddings)

        pooled_embeddings = tf.keras.layers.GlobalMaxPooling1D()(output_embeddings)
        document_embeddings = ExtractEmbeddingsLayer(self.config)([output_embeddings, labels])

        return pooled_embeddings + document_embeddings

    def compute_embeddings(self, inputs):

        encoded_inputs, labels, positions = self.encode_inputs(inputs)

        encoded_inputs = tf.keras.layers.Reshape((-1,))(encoded_inputs)

        positions = tf.keras.layers.Reshape((-1,))(positions)

        labels = tf.keras.layers.Reshape((-1, 1))(labels)
        labels = tf.keras.layers.Masking(mask_value=0)(labels)

        input_embeddings = tf.keras.layers.Embedding(self.get_input_vocab_size(),
            self.get_layer_size() // 2, mask_zero=True)(encoded_inputs)

        hidden = AddPositionEncodingLayer(self.config)([input_embeddings, positions])

        return hidden, labels

    def encode_inputs(self, inputs):
        self.encoder_layer = EncoderLayer(self.config, self.training_dataset)

        return self.encoder_layer(inputs)

    def get_vocab_size(self):
        return self.encoder_layer.get_vocab_size()

    def get_input_vocab_size(self):
        # one for the zero masked value
        # one for the special embedding token
        return self.encoder_layer.get_vocab_size() + 3

    def get_layer_size(self):
        return int(self.config["model"]["layer-size"])

    def get_layer_count(self):
        return int(self.config["model"]["layer-count"])

    def get_epochs(self):
        return int(self.config["model"]["epochs"])

    def get_learning_rate(self):
        return float(self.config["model"]["learning-rate"])

    def get_dropout(self):
        return float(self.config["model"]["dropout"])

    def get_profile_batch(self):
        should_profile = str(self.config["model"]["enable-profiler"]).lower() in ['true', '1']

        if should_profile:
            return 3
        else:
            return False

    def get_best_model_directory(self):
        return os.path.join(self.config['model']['directory'], 'best.h5')

    def get_checkpoint_model_directory(self):
        return os.path.join(self.config['model']['directory'], 'checkpoint.h5')

    def get_permutation_count(self):
        return int(self.config['model']['permutation-count'])

