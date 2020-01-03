
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source'))

from argparse import ArgumentParser

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

from engine.run_locally  import run_locally

def main():

    parser = ArgumentParser(description="A script for training and inference using "
        "the GenderDefender Savant.")

    parser.add_argument("-n", "--experiment-name", default = "",
        help = "A unique name for the experiment.")
    parser.add_argument("-p", "--predict", default = False, action="store_true",
        help = "Run prediction on a specified trained model instead of training.")
    parser.add_argument("-m", "--model-path", default = "configs/sota.json",
        help = "Load the specified model or config file.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")
    parser.add_argument("-L", "--enable-logger", default = [], action="append",
        help = "Enable logging for a specific module")
    parser.add_argument("-O", "--override-config", default = [], action="append",
        help = "Override config file arguments")
    parser.add_argument("--test-set", default="",
        help = "The path to the test set to run on.")
    parser.add_argument("--test-set-size", default=1024,
        help = "The test set size in example count.")
    parser.add_argument("--data-source-type", default="RankingCsvDataSource",
        help = "The type of dataset.")
    parser.add_argument("-o", "--output-directory", default="predictions.csv",
        help = "The output directory the save the output of the inference run.")

    arguments = vars(parser.parse_args())

    run_locally(arguments)

################################################################################
## Guard Main
if __name__ == "__main__":
    main()
################################################################################



