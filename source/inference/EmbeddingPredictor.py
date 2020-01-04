
from model.ModelFactory import ModelFactory

import tensorflow as tf

import csv
import math
import numpy
import sklearn.cluster

class EmbeddingPredictor:
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.model = ModelFactory(config).create()
        self.row_count = 0

    def predict(self):

        self.compute_embeddings()

    def compute_embeddings(self):

        with open(self.get_output_path(), 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"')

            for batch in self.dataset.get_tensorflow_dataset():
                embeddings = self.model.predict_on_batch(batch[0])

                batch_size = batch[0].numpy().shape[0]

                for i in range(batch_size):
                    text = batch[0].numpy()[i].decode('utf8')
                    flat_embeddings = embeddings[i,0,:]

                    row = [flat_embeddings]

                    if self.should_include_text():
                        row = [text] + row

                    writer.writerow(row)

                    self.row_count += 1

                    if self.row_count % 1000 == 0:
                        print("Ran inference on", self.row_count, "batches", end="\r")

    def get_output_path(self):
        return self.config["output_directory"]

    def should_include_text(self):
        if "predictor" in self.config:
            if "should-include-text" in self.config["predictor"]:
                return str(self.config["predictor"]["should-include-text"]).lower() in ['true', '1']

        return True



