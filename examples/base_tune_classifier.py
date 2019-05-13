import subprocess
import os
import sys

lrs = ['2e-5', '3e-5']
bss = ['32']
msls = ['512', '384', '256']
if sys.argv[1] == '0':
    nepochs = ['4']
else:
    nepochs = ['2', '3']
    
if __name__ == '__main__':
    for bs in bss:
        for msl in msls:
            for lr in lrs:
                for nepoch in nepochs:
                    output_dir = './tmp/jigsaw-msl{}-bs{}-lr{}-nepoch{}'.format(msl, bs, lr.replace('-', ''), nepoch)
                    print(output_dir)
                    if os.path.exists(output_dir):
                        print('exists...skip')
                        continue
                    subprocess.call("python run_classifier.py --task_name jigsaw --do_train --do_lower_case         \
                            --data_dir /hdfs/input/xiaguo/ --bert_model bert-base-uncased --max_seq_length {}       \
                            --train_batch_size {} --learning_rate {} --num_train_epochs {}                          \
                            --output_dir {}".format(msl, bs, lr, nepoch, output_dir), shell=True)
