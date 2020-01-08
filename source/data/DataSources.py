
import tensorflow as tf

#AUTOTUNE = tf.data.experimental.AUTOTUNE
AUTOTUNE=16

class DataSources:
    def __init__(self, config):
        self.sources = []
        self.config = config

    def get_tensorflow_dataset(self):

        with tf.device('/cpu:0'):
            dataset = self.sources[0].get_tensorflow_dataset()

            if len(self.sources) > 1:
                for source in self.sources[1:]:
                    dataset = dataset.concatenate(source.get_tensorflow_dataset())

            dataset = dataset.shuffle(self.get_shuffle_window_size())
            dataset = dataset.batch(self.get_mini_batch_size())
            dataset = dataset.prefetch(buffer_size=AUTOTUNE)

            return dataset

    def add_source(self, source):
        self.sources.append(source)

    def get_mini_batch_size(self):
        return int(self.config['model']['batch-size'])

    def get_shuffle_window_size(self):
        return int(self.config['model']['shuffle-window-size'])

    def get_raw_text_datasets(self):
        dataset = self.sources[0].get_raw_text_dataset()

        if len(self.sources) > 1:
            for source in self.sources[1:]:
                dataset = dataset.concatenate(source.get_raw_text_dataset())

        return dataset

    def get_raw_text_generator(self):

        iterator = iter(self.get_raw_text_datasets())

        while True:
            try:
                x = next(iterator).numpy()
                yield x
            except StopIteration:
                return




