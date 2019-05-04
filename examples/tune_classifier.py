import subprocess
import os
import sys

lrs = ['2e-5', '3e-5', '4e-5']
nepochs = ['1', '2', '3']
bss = ['32', '64', '128', '256']
msls = ['128', '196', '256', '512']

if sys.argv[1] == '0':
    msls = ['256', '512']
else:
    msls = ['128', '196']
print(msls)

if __name__ == '__main__':
    for nepoch in nepochs:
        for lr in lrs:
            for msl in msls:
                for bs in bss:
                    output_dir = './tmp/jigsaw-msl{}-bs{}-lr{}-nepoch{}'.format(msl, bs, lr.replace('-', ''), nepoch)
                    print(output_dir)
                    if os.path.exists(output_dir):
                        print('exists...skip')
                        continue
                    subprocess.call("python run_classifier.py --task_name jigsaw --do_train --do_lower_case         \
                            --data_dir /hdfs/input/xiaguo/kgjs19 --bert_model bert-base-uncased --max_seq_length {} \
                            --train_batch_size {} --learning_rate {} --num_train_epochs {}                          \
                            --output_dir {}".format(msl, bs, lr, nepoch, output_dir), shell=True)
