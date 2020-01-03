
from inference.ClusterPredictor import ClusterPredictor

class PredictorFactory:
    def __init__(self, config):
        self.config = config

    def create(self):
        if self.config["predictor"]["type"] == "ClusterPredictor":
            return ClusterPredictor(self.config)


        assert False



