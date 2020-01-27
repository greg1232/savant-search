
import tensorflow as tf
import math

AUTOTUNE = tf.data.experimental.AUTOTUNE

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
            dataset = self.group_by_sequence_length(dataset)
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

    def group_by_sequence_length(self, dataset):

        def get_length(x, y):
            return tf.strings.length(x[0])

        boundaries = self.get_bucket_boundaries()

        bucket_transformation = tf.data.experimental.bucket_by_sequence_length(
            element_length_func = get_length,
            bucket_boundaries = boundaries,
            bucket_batch_sizes = [self.get_mini_batch_size() for i in range(len(boundaries) + 1)],
            padded_shapes=None,
            padding_values=None,
            pad_to_bucket_boundary=False,
            no_padding=True,
            drop_remainder=True)

        return dataset.apply(bucket_transformation)

    def get_bucket_boundaries(self):
        base = 2

        bucket_count = int(math.ceil(math.log(self.get_maximum_sequence_length(), base)))

        return [base ** i for i in range(1, bucket_count)]

    def get_maximum_sequence_length(self):
        return int(self.config['model']['maximum-sequence-length'])




