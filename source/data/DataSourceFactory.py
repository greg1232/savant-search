
from data.RankingCsvDataSource import RankingCsvDataSource

class DataSourceFactory:
    def __init__(self, config):
        self.config = config

    def create(self, source_description):

        if source_description["type"] == "RankingCsvDataSource":
            return RankingCsvDataSource(self.config, source_description)

        raise RuntimeError("Unknown data source type '" + source_description["type"] + "'")







