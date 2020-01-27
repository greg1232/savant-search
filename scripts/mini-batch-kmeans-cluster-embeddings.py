from argparse import ArgumentParser
import logging
import csv
import numpy
import random

from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("This program clusters a set of embeddings using the "
    " mini-batch kmeans algorithm.")

    parser.add_argument("-i", "--input-path", default="",
        help = "The path to input csv file.")
    parser.add_argument("-o", "--output-path", default="",
        help = "The path to output csv file.")
    parser.add_argument("-m", "--maximum-samples", default=10000,
        help = "The maximum number of documents to use.")
    parser.add_argument("-c", "--clusters", default=10,
        help = "The number of clusters to use.")
    parser.add_argument("-b", "--batch-size", default=256,
        help = "The minibatch size.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    assign_clusters(arguments)

def assign_clusters(arguments):

    kmeans = MiniBatchKMeans(n_clusters=int(arguments["clusters"]), random_state=0,
        batch_size=int(arguments["batch_size"]))

    for rows, batch in get_batches(arguments):
        kmeans.partial_fit(batch)

    with open(arguments["output_path"], "w") as output_file:
        writer = csv.writer(output_file, delimiter=',', quotechar='"')

        for rows, batch in get_batches(arguments):
            clusters = kmeans.predict(batch)

            for i in range(len(rows)):
                row = rows[i]
                cluster = clusters[i]

                row.append(cluster)

                writer.writerow(row)


def get_batches(arguments):

    with open(arguments["input_path"], "r") as input_file:
        reader = csv.reader(input_file, delimiter=',', quotechar='"')

        row_batch = []
        embedding_batch = []
        count = 0

        for row in reader:
            embedding = numpy.fromstring(row[-1][1:-1], sep=' ')

            row_batch.append(row)
            embedding_batch.append(embedding)

            if len(row_batch) >= int(arguments["batch_size"]):
                new_row_batch = row_batch
                new_embedding_batch = embedding_batch

                row_batch = []
                embedding_batch = []

                count += len(row_batch)

                yield new_row_batch, numpy.stack(new_embedding_batch, axis=0)

            if count >= int(arguments["maximum_samples"]):
                break

def setup_logger(arguments):

   if arguments["verbose"]:
       logger.setLevel(logging.DEBUG)
   else:
       logger.setLevel(logging.INFO)

   ch = logging.StreamHandler()
   ch.setLevel(logging.DEBUG)

   # create formatter
   formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

   # add formatter to ch
   ch.setFormatter(formatter)

   # add ch to logger
   logger.addHandler(ch)

main()

