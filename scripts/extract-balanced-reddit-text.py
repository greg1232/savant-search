

from argparse import ArgumentParser
import logging
import csv
import json

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("This program extracts lines of text from a "
        "reddit data json evenly among subreddits.")

    parser.add_argument("-i", "--input-path", default = "",
        help = "The path to search for text.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")
    parser.add_argument("-L", "--enable-logger", default = [], action="append",
        help = "Enable logging for a specific module.")
    parser.add_argument("-o", "--output-path", default = "",
        help = "Set the output path to save the labels.")
    parser.add_argument("-s", "--subreddits", default = [], action="append",
        help = "Select specific subreddits.")
    parser.add_argument("-c", "--maximum-extracted-count-per-subreddit", default = 1e4,
        help = "Set the maximum number of comments to extract.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    extract_text(arguments)

def extract_text(arguments):

    counter = 0
    subreddits = {subreddit: 0 for subreddit in arguments["subreddits"]}

    maximum_rows = int(arguments["maximum_extracted_count_per_subreddit"]) * len(subreddits)

    with open(arguments["input_path"], "r") as input_file:
        with open(arguments["output_path"], "w") as output_file:
            writer = csv.writer(output_file, delimiter=',', quotechar='"')
            for line in input_file:
                json_string = json.loads(line)
                subreddit = json_string["subreddit"]

                if not subreddit in subreddits:
                    continue

                if subreddits[subreddit] >= int(arguments["maximum_extracted_count_per_subreddit"]):
                    continue

                subreddits[subreddit] += 1

                text = json_string["body"]

                if len(text) == 0:
                    continue

                score = json_string["score"]

                writer.writerow([text, subreddit, score, 1200.0])
                counter += 1

                if counter > maximum_rows:
                    break

    logger.info(str(subreddits))

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


