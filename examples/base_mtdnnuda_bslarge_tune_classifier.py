import subprocess
import os
import logging
import sys

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

lrs = ['2e-5', '3e-5']
ns = [4]
bss = [256, 512, 1024]
seeds = [42]
msls = [360]
tasks = ['jigsaw-u-s']

data_dir = '/hdfs/input/xiaguo/'

if __name__ == '__main__':
    params = []
    for seed in seeds:
        for lr in lrs:
            for n in ns:
                params.append((seed, lr, n))

    tm = []
    for bs in bss:
        for msl in msls:
            for task in tasks:
                tm.append((task, msl, bs))

    rank = int(sys.argv[1])
    task, msl, bs = tm[rank]

    for seed, lr, n in params:
        name = 'js_mtdnnbaseuda_{}_msl{}_bs{}_lr{}_n{}_sd{}'.format(task, msl, bs, lr.replace('-', ''), n, seed)
        logger.info('Processing - {}'.format(name))

        output_dir =  os.path.join('.', 'jigsaw-out', name)
        output_file = '{}.csv'.format(name)

        training = True

        # If exists, skipping...
        if os.path.exists(os.path.join(output_dir, 'config.json')):
            logger.info('{} exists, skipping training...'.format(output_dir))
            training = False

        if training:
            mtdnn_bert_dir = '/hdfs/input/xiaguo/mtdnnbert_base_uncased'
            if os.path.exists(os.path.join(output_dir)):
                for e in range(n - 1, -1, -1):
                    subdir = os.path.join(output_dir, 'epoch{}'.format(e))
                    if os.path.exists(os.path.join(subdir, 'config.json')):
                        mtdnn_bert_dir = subdir
                        logger.info('training continued from {}'.format(mtdnn_bert_dir))
                        break

            # Train toxic classifier
            assert bs % 32 == 0
            gas = bs // 32
            ret = subprocess.call("python train_jigsaw19_uda.py --task_name {} --data_dir {}    \
                --bert_model {} --max_seq_length {} --gradient_accumulation_steps {}            \
                --train_batch_size {} --learning_rate {} --num_train_epochs {} --seed {}        \
                --output_dir {} --feature_cache_dir {} --use_feature_cache".format(
                    task, data_dir, mtdnn_bert_dir, msl, gas, bs, lr, n, seed, output_dir, './feature_cache_mtdnn'), shell=True)

            if ret != 0:
                logger.error("Error train classifier, exit")
                sys.exit(ret)

        infer_output_dir = os.path.join(output_dir, 'results')

        # Infer toxic comments
        command = "python infer_jigsaw19.py --do_lower_case --data_dir {} --task_name {} \
            --max_seq_length {} --infer_batch_size 64".format(data_dir, task, msl)
        
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
