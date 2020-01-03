
from model.ModelFactory import ModelFactory

import tensorflow as tf

import csv
import math
import numpy
import sklearn.cluster

class ClusterPredictor:
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.model = ModelFactory(config).create()
        self.row_count = 0

    def predict(self):

        all_embeddings = self.compute_embeddings()

        clusters, cluster_centers = self.cluster_embeddings(all_embeddings)

        self.write_clusters(clusters, cluster_centers)

    def compute_embeddings(self):
        dataset_embeddings = []

        for batch in self.dataset.get_tensorflow_dataset():
            embeddings = self.model.predict_on_batch(batch[0])

            dataset_embeddings.append((batch[0], embeddings))

            self.row_count += 1

            if self.row_count % 1000 == 0:
                print("Ran inference on", self.row_count, "batches", end="\r")

        return dataset_embeddings

    def cluster_embeddings(self, embeddings):

        embeddings_data = [embedding[i, 0, :] for text, embedding in embeddings for i in range(embedding.shape[0])]

        cluster_model = sklearn.cluster.KMeans(
            n_clusters=self.get_cluster_count(), random_state=0)

        cluster_model.fit(embeddings_data)

        embedding_clusters = cluster_model.predict(embeddings_data)

        embeddings_and_clusters = [(embedding[0], embedding[1], cluster) for
            embedding, cluster in zip(embeddings, embedding_clusters)]

        cluster_centers = cluster_model.cluster_centers_

        return embeddings_and_clusters, cluster_centers

    def write_clusters(self, clusters, cluster_centers):

        with open(self.get_output_path(), 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"')

            for text, embedding, cluster in sorted(clusters, key=lambda x: x[2]):
                row = [embedding, cluster]

                if self.should_include_text():
                    row = [text] + row

                writer.writerow(row)

    def get_output_path(self):
        return self.config["output_directory"]

    def get_cluster_count(self):
        return int(self.config["predictor"]["cluster-count"])

    def should_include_text(self):
        if "predictor" in self.config:
            return str(self.config["predictor"]["should-include-text"]).lower() in ['true', '1']

        return True


