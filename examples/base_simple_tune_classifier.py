import subprocess
import os
import sys

lrs = ['2e-5', '3e-5']
ns = [3]
bss = [64]
seeds = [8, 6]
if sys.argv[1] == '0':
    msl = 256
    task = 'jigsaw-r'
    cls_model = 'default'
elif sys.argv[1] == '1':
    msl = 384
    task = 'jigsaw-r'
    cls_model = 'default'
elif sys.argv[1] == '2':
    msl = 384
    task = 'jigsaw-b'
    cls_model = 'default'
elif sys.argv[1] == '3':
    msl = 384
    task = 'jigsaw-b'
    cls_model = 'context'
elif sys.argv[1] == '4':
    msl = 256
    task = 'jigsaw-r'
    cls_model = 'default'
elif sys.argv[1] == '5':
    msl = 384
    task = 'jigsaw-r'
    cls_model = 'default'
else:
    raise ValueError("Not recognized argv")

if __name__ == '__main__':
    params = []
    for seed in seeds:
        for lr in lrs:
            for bs in bss:
                for n in ns:
                    params.append((seed, lr, bs, n))

    for seed, lr, bs, n in params:
        name = 'js_base_{}_msl{}_bs{}_lr{}_n{}_seed{}_cls{}'.format(task, msl, bs, lr.replace('-', ''), n, seed, cls_model)
        output_dir = './tmp/{}'.format(name)
        output_file = '{}.csv'.format(name)
        with open("{}.txt".format(sys.argv[1]), "a") as f:
            print(output_dir)
            f.write(output_dir)
            if os.path.exists(output_dir):
                print('exists...skip')
                f.write(' - exists...skip')
                f.write('\n')
                continue
            f.write('\n')
        subprocess.call("python train_jigsaw19.py --task_name {} \
            --data_dir /hdfs/input/xiaguo/ --bert_model bert-base-uncased --max_seq_length {}   \
            --train_batch_size {} --learning_rate {} --num_train_epochs {} --seed {}            \
            --output_dir {} --feature_cache_dir {} --use_feature_cache --cls_model {}".format(
                task, msl, bs, lr, n, seed, output_dir, './feature_cache', cls_model), shell=True)
        subprocess.call("python infer_jigsaw19.py --do_lower_case --data_dir /hdfs/input/xiaguo \
            --bert_model {} --max_seq_length {} --output_dir {} --output_file {} --task_name {} \
            --infer_batch_size 64".format(output_dir, msl, output_dir, output_file, task), shell=True)
        for subdir in [f.name for f in os.scandir(output_dir) if f.is_dir()]:
            output_subdir = os.path.join(output_dir, subdir)
            output_subfile = '{}-{}.csv'.format(name, subdir)
            subprocess.call("python infer_jigsaw19.py --do_lower_case --data_dir /hdfs/input/xiaguo \
                --bert_model {} --max_seq_length {} --output_dir {} --output_file {} --task_name {} \
                --infer_batch_size 64".format(output_subdir, msl, output_subdir, output_subfile, task), shell=True)
