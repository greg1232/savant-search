
import os

import tensorflow as tf

from model.SimpleSequenceEmbeddingModel import SimpleSequenceEmbeddingModel
from model.SimpleAttentionModel import SimpleAttentionModel

class ModelFactory:
    def __init__(self, config, *, training_data=None, validation_data=None):

        self.config = config
        self.model_name = config["model"]["type"]
        self.validation_data = validation_data
        self.training_data = training_data

    def create(self):

        if self.model_name == "SimpleSequenceEmbeddingModel":
            return SimpleSequenceEmbeddingModel(self.config,
                self.training_data, self.validation_data)

        if self.model_name == "SimpleAttentionModel":
            return SimpleAttentionModel(self.config,
                self.training_data, self.validation_data)

        raise RuntimeError("Unknown model name " + self.model_name)




