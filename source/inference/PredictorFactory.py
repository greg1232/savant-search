
from inference.ClusterPredictor import ClusterPredictor
from inference.EmbeddingPredictor import EmbeddingPredictor

class PredictorFactory:
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

    def create(self):
        if self.config["predictor"]["type"] == "ClusterPredictor":
            return ClusterPredictor(self.config, self.dataset)

        if self.config["predictor"]["type"] == "EmbeddingPredictor":
            return EmbeddingPredictor(self.config, self.dataset)

        assert False



