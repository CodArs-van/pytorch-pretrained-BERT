import argparse
import subprocess
import os
import logging
import sys

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--bert_model", type=str,
                    help="The path of pretrained bert model.")
args = parser.parse_args()

if __name__ == "__main__":
    data_dir = '/hdfs/input/xiaguo/'
    task = 'jigsaw-b-s'
    msl = 512
    output_dir = 'aug-jigsaw'
    bs = 64
    seed = 42
    bert_model = args.bert_model
    output_name = "{}.csv".format(os.path.basename(bert_model))
    logger.info("task: {}, msl: {}, bs: {}, seed: {}".format(task, msl, bs, seed))
    logger.info("data_dir: {}".format(data_dir))
    logger.info("output_dir: {}".format(output_dir))
    logger.info("output_name: {}".format(output_name))
    logger.info("bert_model: {}".format(bert_model))

    ret = subprocess.call("python infer_aug_processor.py --task_name {} --data_dir {}      \
        --bert_model {} --max_seq_length {} --output_name {} --do_lower_case \
        --infer_batch_size {} --seed {} --output_dir {}".format(
            task, data_dir, bert_model, msl, output_name, bs, seed, output_dir), shell=True)

    if ret != 0:
        logger.error("Error train classifier, exit")
        sys.exit(ret)
