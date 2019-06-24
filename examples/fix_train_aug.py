import argparse
import logging
import math
import os
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, type=str, help="path to train_aug.csv")
args = parser.parse_args()

if __name__ == "__main__":
    df = pd.read_csv(args.data)
    for i, (text_a, text_b, isnan) in enumerate(zip(df['comment_text'], df['comment_text_ori'], df['comment_text'].isnull())):
        if isnan:
            logger.info("{} - text_a: {} - text_b: {}".format(i, text_a, text_b))
            # df['comment_text'][i] = text_b
    # df.to_csv('train_aug_fix.csv', index=False)