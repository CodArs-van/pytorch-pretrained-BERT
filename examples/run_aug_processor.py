import subprocess
import os
import logging
import sys

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

data_dir = '/hdfs/input/xiaguo/'
task = 'jigsaw-b-s'
msl = 512
bs = 512

mini = 64 if msl <= 384 else 32
assert bs % mini == 0
gas = bs // mini

output_dir =  os.path.join('.', 'aug-jigsaw-out')
lr = 5e-5
n = 10
seed = 42

if __name__ == '__main__':
    logger.info('data_dir: {}'.format(data_dir))
    logger.info('msl: {}, bs: {}, gas: {}, lr: {}, n: {}, seed: {}'.format(
        msl, bs, gas, lr, n, seed))
    ret = subprocess.call("python train_jigsaw19.py --task_name {} --data_dir {}            \
        --bert_model bert-base-uncased --max_seq_length {} --gradient_accumulation_steps {} \
        --train_batch_size {} --learning_rate {} --num_train_epochs {} --seed {}            \
        --output_dir {} --do_lower_case".format(
            task, data_dir, msl, gas, bs, lr, n, seed, output_dir), shell=True)