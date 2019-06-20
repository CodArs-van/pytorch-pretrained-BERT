import subprocess
import os
import logging
import re
import sys

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# msl, bs, lr, seed
tune_params = [
    ('js_base_jigsaw-b-s_msl512_bs512_lr2e5_n3_sd77-e2-idx4', 'kgjs19_77', '0.94148'), # wu1
    ('js_base_jigsaw-b-s_msl512_bs512_lr2e5_n3_sd42-e2-idx4', 'kgjs19_42', '0.94147'), # wu1
    ('js_base_jigsaw-b-s_msl384_bs512_lr3e5_n3_sd88-e1-idx9', 'kgjs19_88', '0.94172'), # wu1
    ('js_base_jigsaw-b-s_msl280_bs512_lr3e5_n4_sd42-epoch1', 'kgjs19_42', '0.94173'), # wu1
    ('js_base_jigsaw-b-s_msl360_bs512_lr3e5_n4_sd42-epoech1', 'kgjs19_42', '0.94208'), # sc2
    ('js_mtdnnbase_jigsaw-b-s_msl360_bs128_lr3e5_n3_sd42-epoch1', 'kgjs19_42', '0.94135'), # sc2
    ('js_mtdnnbase_jigsaw-b-s_msl360_bs128_lr2e5_n3_sd42-epoch1', 'kgjs19_42', '0.94149'), # sc2
    ('js_mtdnnbase_jigsaw-b-s_msl360_bs256_lr2e5_n4_sd42-epoch1', 'kgjs19_42', '0.94128'),
    ('js_mtdnnbase_jigsaw-b-s_msl280_bs128_lr3e5_n3_sd42-epoch1', 'kgjs19_42', '0.94123'), # sc2
    ('js_base_jigsaw-b-s_msl384_bs128_lr2e5_n3_sd42-epoch1', 'kgjs19_42', '0.94206'), # sc2
    ('js_mtdnnbase_jigsaw-b-s_msl280_bs128_lr2e5_n3_sd42-epoch1', 'kgjs19_42', '0.94141'), # sc2
    ('js_base_jigsaw-b-s_msl512_bs512_lr3e5_n3_sd42/e2-idx4', 'kgjs19_42', '0.94140'), # sc2
]
task = 'jigsaw-b-s'
n = 3
lr = '1e-5'

data_dir = '/hdfs/input/xiaguo'

if __name__ == '__main__':
    params = [1]
    rank = int(sys.argv[1])
    fname, subfolder, score = tune_params[rank]
    logger.info('Score: {}'.format(score))
    logger.info('lr: {}, task: {}, n: {}'.format(lr, task, n))

    m = re.search(r'sd(\d+)', fname)
    seed = m.group(1)
    dirname = fname[:m.end()]
    subdirname = fname[m.end() + 1:]
    bert_model = '{}/{}/{}/{}'.format(data_dir, subfolder, dirname, subdirname)
    msl = re.search(r'_msl(\d+)', fname).group(1)
    bs = int(re.search(r'_bs(\d+)', fname).group(1))
    logger.info('msl: {}, bs: {}, seed: {}'.format(msl, bs, seed))

    logger.info('bert_model: {}'.format(bert_model))
    if not os.path.exists(bert_model):
        logger.error('bert_model not exists!')
        sys.exit(-1)

    for _ in params:
        name = 'js_lowlr_{}_msl{}_bs{}_lr{}_n{}_sd{}'.format(task, msl, bs, lr.replace('-', ''), n, seed)
        logger.info('Processing - {}'.format(name))

        output_dir =  os.path.join('.', 'jigsaw-out', name)
        output_file = '{}.csv'.format(name)

        training = True

        # If exists, skipping...
        if os.path.exists(output_dir):
            logger.info('{} exists, skipping training...'.format(output_dir))
            training = False

        if training:
            mini = 64 if bs <= 384 else 32
            assert bs % mini == 0
            gas = bs // mini
            # Train toxic classifier
            ret = subprocess.call("python train_jigsaw19_lowlr.py --task_name {} --data_dir {}      \
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
