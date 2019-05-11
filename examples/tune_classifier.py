import subprocess
import os
import sys

lrs = ['2e-5', '3e-5', '5e-5']
nepochs = ['2', '1', '3']
bss = ['16', '32', '64']
msls = ['256', '384']

if sys.argv[1] == '0':
    msls = ['256', '384']
else:
    msls = ['512']
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
                            --data_dir /hdfs/input/xiaguo/ --bert_model bert-base-uncased --max_seq_length {}       \
                            --train_batch_size {} --learning_rate {} --num_train_epochs {}                          \
                            --output_dir {}".format(msl, bs, lr, nepoch, output_dir), shell=True)
