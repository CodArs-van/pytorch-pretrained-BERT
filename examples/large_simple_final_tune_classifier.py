import subprocess
import os
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# msl, bs, lr, seed
tune_params = [
    (512, 512, '2e-5', 42),
    (512, 512, '3e-5', 42),
]
task = 'jigsaw-b-s'
n = 2.5

data_dir = '/hdfs/input/xiaguo/'
bert_model = '/hdfs/input/xiaguo/bert-large-uncased-whole-word-masking'

if __name__ == '__main__':
    params = [1]
    rank = int(sys.argv[1])
    msl, bs, lr, seed = tune_params[rank]

    for _ in params:
        name = 'js_large_{}_msl{}_bs{}_lr{}_n{}_sd{}'.format(task, msl, bs, lr.replace('-', ''), n, seed)
        logger.info('Processing - {}'.format(name))

        output_dir =  os.path.join('.', 'jigsaw-out', name)
        output_file = '{}.csv'.format(name)

        training = True

        # If exists, skipping...
        if os.path.exists(output_dir):
            logger.info('{} exists, skipping training...'.format(output_dir))
            training = False

        if training:
            mini = 8
            assert bs % mini == 0
            gas = bs // mini
            # Train toxic classifier
            ret = subprocess.call("python train_jigsaw19.py --task_name {} --data_dir {}            \
                --bert_model {} --max_seq_length {} --gradient_accumulation_steps {} \
                --train_batch_size {} --learning_rate {} --num_train_epochs {} --seed {}            \
                --output_dir {} --feature_cache_dir {} --use_feature_cache".format(
                    task, data_dir, bert_model, msl, gas, bs, lr, n, seed, output_dir, './feature_cache'), shell=True)

            if ret != 0:
                logger.error("Error train classifier, exit")
                sys.exit(ret)

        infer_output_dir = os.path.join(output_dir, 'results')

        # Infer toxic comments
        command = "python infer_jigsaw19.py --do_lower_case --data_dir {} --task_name {} \
            --max_seq_length {} --infer_batch_size 16".format(data_dir, task, msl)
        
        if not os.path.exists(os.path.join(infer_output_dir, output_file)):
            ret = subprocess.call("{} --bert_model {} --output_dir {} --output_file {}".format(
                command, output_dir, infer_output_dir, output_file), shell=True)

            if ret != 0:
                logger.error("Error infer main model")
                sys.exit(ret)
        
        for subdir in [f.name for f in os.scandir(output_dir) if f.is_dir()]:
            output_subdir = os.path.join(output_dir, subdir)
            output_subfile = '{}-{}.csv'.format(name, subdir)

            if os.path.exists(os.path.join(infer_output_dir, output_subfile)):
                logger.info('{} already inferenced...Skip...'.format(output_subdir))
                continue

            if not os.path.exists(os.path.join(output_subdir, 'config.json')):
                logger.info('Skipping {}'.format(output_subdir))
                continue

            ret = subprocess.call("{} --bert_model {} --output_dir {} --output_file {}".format(
                command, output_subdir, infer_output_dir, output_subfile), shell=True)

            if ret != 0:
                logger.error("Error infer sub model")
                sys.exit(ret)
