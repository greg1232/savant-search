
import numpy
import csv
import os

import tensorflow as tf

class RankingCsvDataSource:
    def __init__(self, config, source_config):
        self.config = config
        self.source_config = source_config
        self.files = self.get_files()

    def get_files(self):

        if os.path.isfile(self.get_path()):
            return [self.get_path()]

        allFiles = []

        for root, directories, files in os.walk(self.get_path()):
            allFiles += [os.path.join(root, f) for f in files]

        return sorted(all_files)

    def get_raw_text_dataset(self):
        line_dataset = tf.data.experimental.CsvDataset(self.files, [tf.string, tf.float32])

        raw_text_dataset = line_dataset.map(lambda x, y : x)

        return raw_text_dataset

    def get_tensorflow_dataset(self):
        line_dataset = tf.data.experimental.CsvDataset(self.files, [tf.string, tf.float32])

        if self.has_maximum_size():
            line_dataset = line_dataset.take(self.get_maximum_size())

        text_dataset = line_dataset.map(lambda x, y : self.load_and_tokenize((x,y)))

        return text_dataset

    def load_and_tokenize(self, row):

        text_class = tf.math.greater(row[1], 1200.0)

        if self.get_should_invert_class():
            negative = text_class
            positive = tf.math.logical_not(text_class)
        else:
            negative = tf.math.logical_not(text_class)
            positive = text_class

        return row[0], [negative, positive]

    def get_path(self):
        return self.source_config['path']

    def get_should_invert_class(self):
        return bool(self.config["model"]["invert-labels"])

    def get_maximum_size(self):
        return int(self.source_config["maximum-size"])

    def has_maximum_size(self):
        return "maximum-size" in self.source_config






