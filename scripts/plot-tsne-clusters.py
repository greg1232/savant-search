from argparse import ArgumentParser
import logging
import csv
import json
import matplotlib.pyplot
import numpy
import random

import matplotlib.cm as cm

from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("This program plots the tsne for embeddings.")

    parser.add_argument("-i", "--input-path", default="",
        help = "The path to input csv file.")
    parser.add_argument("-m", "--maximum-samples", default=10000,
        help = "The maximum number of documents to use.")
    parser.add_argument("-c", "--maximum-clusters", default=10,
        help = "The maximum  number of clusters to use.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    plot_tsne_for_csv(arguments)

def plot_tsne_for_csv(arguments):
    matplotlib.pyplot.figure()

    embeddings, clusters = load_csv(arguments)

    all_tsne_embeddings = TSNE().fit_transform(embeddings)

    tsne_embeddings_per_cluster = separate_clusters(clusters, all_tsne_embeddings)

    colors = cm.rainbow(numpy.linspace(0, 1, len(tsne_embeddings_per_cluster)))

    for cluster, tsne_embeddings in sorted(tsne_embeddings_per_cluster.items(), key=lambda x:x[0]):
        matplotlib.pyplot.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1],
            label=str(cluster), color=colors[cluster])

    #matplotlib.pyplot.ylim([0.0, 1.05])
    matplotlib.pyplot.xlabel('First Dimension')
    matplotlib.pyplot.ylabel('Second Dimension')
    matplotlib.pyplot.title('TSNE Embeddings')
    matplotlib.pyplot.legend(loc="lower right")

    matplotlib.pyplot.show()

def separate_clusters(clusters, embeddings):
    cluster_embeddings = {}

    for i, cluster in enumerate(clusters):
        if not cluster in cluster_embeddings:
            cluster_embeddings[cluster] = []

        cluster_embeddings[cluster].append(embeddings[i,:])

    return {cluster : numpy.array(cluster_embeddings[cluster]) for cluster in cluster_embeddings.keys()}

def load_csv(arguments):
    embeddings_and_clusters = []

    with open(arguments["input_path"], "r") as input_file:
        reader = csv.reader(input_file, delimiter=',', quotechar='"')
        for row in reader:
            embedding = numpy.fromstring(row[-3][1:-1], sep=' ').tolist()
            cluster = int(row[-1])

            if cluster >= int(arguments["maximum_clusters"]):
                continue

            embeddings_and_clusters.append((embedding, cluster))

    generator = random.Random(0)

    generator.shuffle(embeddings_and_clusters)

    length = min(len(embeddings_and_clusters), int(arguments["maximum_samples"]))

    embeddings_and_clusters = embeddings_and_clusters[0:length]

    embeddings = [embeddings for embeddings, clusters in embeddings_and_clusters]
    clusters = [clusters for embeddings, clusters in embeddings_and_clusters]

    return embeddings, clusters

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



