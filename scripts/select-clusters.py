
from argparse import ArgumentParser
import logging
import csv
import json
import numpy

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("This program selects training data belonging to a specified cluster.")

    parser.add_argument("-i", "--input-path", default = "",
        help = "The path to a csv file with clusters results.")
    parser.add_argument("-c", "--clusters", default = [], action="append",
        help = "The cluster number to select.")
    parser.add_argument("-o", "--output-path", default = "",
        help = "The path to a csv to save the embeddings.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    select_data_from_clusters(arguments)

def select_data_from_clusters(arguments):
    clusters = set([int(cluster) for cluster in arguments["clusters"]])

    with open(arguments["input_path"], "r") as input_file, \
        open(arguments["output_path"], "w") as output_file:
        reader = csv.reader(input_file, delimiter=',', quotechar='"')
        writer = csv.writer(output_file, delimiter=',', quotechar='"')

        for row in reader:
            cluster = int(row[-1])

            if cluster in clusters:
                writer.writerow(row)

def get_cluster(embedding, clusters):
    return numpy.argmin(numpy.linalg.norm(
        numpy.reshape(embedding, (1, embedding.shape[0])) - clusters,
        axis=1))

def load_clusters(arguments):
    cluster_centers = {}

    with open(arguments["cluster_path"], "r") as input_file:
        reader = csv.reader(input_file, delimiter=',', quotechar='"')

        for row in reader:
            cluster = int(row[-1])
            embedding_string = row[-3][1:-1]
            cluster_center = numpy.fromstring(embedding_string, sep=' ').tolist()

            cluster_centers[cluster] = cluster_center

    return numpy.array([center for cluster, center in sorted(cluster_centers.items())])

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

















