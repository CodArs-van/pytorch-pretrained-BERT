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
parser.add_argument("--split_dir", type=str, required=True, help="path that has split train data")
parser.add_argument("--mode", type=str, required=True, help="split/merge")
parser.add_argument("--segment", type=int, required=True, help="How many segment?")
args = parser.parse_args()

data_dir = "/hdfs/input/xiaguo"

if __name__ == "__main__":

    if args.mode == "split":
        df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        rows = df.shape[0]
        ct = rows // args.segment
        logger.info("Total rows: {}".format(rows))
        logger.info("Segment: {}".format(args.segment))
        logger.info("Number of rows for each segment: {}".format(ct))
        for i in range(args.segment):
            start = i * ct
            end = min(start + ct, rows)
            dfsub = df.iloc[start:end]
            distdir = os.path.join(args.split_dir, "seg{}".format(i))
            logger.info("Path distdir: {}".format(distdir))
            os.makedirs(distdir, exist_ok=True)
            dfsub.to_csv(os.path.join(distdir, "train.csv"), index=False)
        logger.info("Done.")
    elif args.mode == "merge":
        dfs = []
        for i in range(args.segment):
            path = os.path.join(args.split_dir, "seg{}".format(i), "train_aug.csv")
            logger.info("Path: {}".format(path))
            dfs.append(pd.read_csv(path))
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(os.path.join(args.split_dir, "train_merge.csv"), index=False)
    else:
        raise ValueError()