import subprocess
import os
import sys

lrs = ['2e-5', '3e-5']
ns = [2, 3]
bss = [32, 64]
seeds = [7, 8, 6]
if sys.argv[1] == '0':
    msl = 256
    md = 'uncased'
    dlc = '--do_lower_case'
    task = 'jigsaw'
elif sys.argv[1] == '1':
    msl = 384
    md = 'uncased'
    dlc = '--do_lower_case'
    task = 'jigsaw'
elif sys.argv[1] == '2':
    msl = 320
    md = 'uncased'
    dlc = '--do_lower_case'
    bss = [64]
    task = 'jigsaw-cls'
elif sys.argv[1] == '3':
    msl = 256
    md = 'cased'
    dlc = ''
    task = 'jigsaw'
elif sys.argv[1] == '4':
    msl = 384
    md = 'cased'
    dlc = ''
    task = 'jigsaw'
elif sys.argv[1] == '5':
    msl = 320
    md = 'uncased'
    dlc = '--do_lower_case'
    bss = [32]
    task = 'jigsaw-cls'
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
        output_dir = './tmp/js-base-msl{}-bs{}-lr{}-n{}-seed{}-md{}'.format(msl, bs, lr.replace('-', ''), n, seed, md)
        output_file = 'js-base-msl{}-bs{}-lr{}-n{}-seed{}-md{}.csv'.format(msl, bs, lr.replace('-', ''), n, seed, md)
        with open("{}.txt".format(sys.argv[1]), "a") as f:
            print(output_dir)
            f.write(output_dir)
            if os.path.exists(output_dir):
                print('exists...skip')
                f.write(' - exists...skip')
                f.write('\n')
                continue
            f.write('\n')
        subprocess.call("python run_classifier.py --task_name {} --do_train {}              \
            --data_dir /hdfs/input/xiaguo/ --bert_model bert-base-{} --max_seq_length {}    \
            --train_batch_size {} --learning_rate {} --num_train_epochs {} --seed {}        \
            --output_dir {}".format(task, dlc, md, msl, bs, lr, n, seed, output_dir), shell=True)
        subprocess.call("python infer_jigsaw19.py {} --data_dir /hdfs/input/xiaguo/         \
            --bert_model {} --max_seq_length {} --output_dir {} --output_file {}            \
            --infer_batch_size 64".format(dlc, output_dir, msl, output_dir, output_file), shell=True)
