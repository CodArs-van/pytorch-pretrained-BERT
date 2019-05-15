import subprocess
import os
import sys

lrs = ['2e-5', '3e-5']
ns = [2, 3]
bss = [32, 64]
seeds = [7, 8, 6]
if sys.argv[1] == '0':
    msl = 256
elif sys.argv[1] == '1':
    msl = 384
elif sys.argv[1] == '2':
    msl = 512

if __name__ == '__main__':
    params = []
    for seed in seeds:
        for lr in lrs:
            for bs in bss:
                for n in ns:
                    params.append((seed, lr, bs, n))

    for seed, lr, bs, n in params:
        output_dir = './tmp/js-base-msl{}-bs{}-lr{}-n{}-seed{}'.format(msl, bs, lr.replace('-', ''), n, seed)
        with open("{}.txt".format(sys.argv[1]), "a") as f:
            print(output_dir)
            f.write(output_dir)
            if os.path.exists(output_dir):
                print('exists...skip')
                f.write(' - exists...skip')
                f.write('\n')
                continue
            f.write('\n')
        subprocess.call("python run_classifier.py --task_name jigsaw --do_train --do_lower_case     \
            --data_dir /hdfs/input/xiaguo/ --bert_model bert-base-uncased --max_seq_length {}       \
            --train_batch_size {} --learning_rate {} --num_train_epochs {} --seed {}                \
            --output_dir {}".format(msl, bs, lr, n, seed, output_dir), shell=True)
