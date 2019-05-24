import subprocess
import os
import sys

lrs = ['2e5', '3e5', '4e5']
nepochs = ['1', '2']
bss = ['32', '64']
msls = ['256', '512']

if __name__ == '__main__':
    for nepoch in nepochs:
        for lr in lrs:
            for msl in msls:
                for bs in bss:
                    data_root = 'C:\\Users\\xiaguo\\Developer\\Philly\\Jigsaw2019'
                    full_name = 'jigsaw-msl{}-bs{}-lr{}-nepoch{}'.format(msl, bs, lr, nepoch)
                    bert_model = '{}\\kgjs19\\{}'.format(data_root, full_name)
                    # if bert_model doesn't exist, skip
                    if not os.path.exists(bert_model):
                        continue
                    # if output file already exists, skip
                    code_root = 'C:\\Users\\xiaguo\\Developer\\PyTorch\\CodArs-pytorch-pretrained-BERT'
                    output_dir = '{}\\.tmp\\jigsaw_output'.format(code_root)
                    output_file_full_path = os.path.join(output_dir, '{}.csv'.format(full_name))
                    if os.path.exists(output_file_full_path):
                        continue
                    print('bert_model: {}'.format(bert_model))
                    print('output_file: {}'.format(output_file_full_path))
                    code_file = '{}\\examples\\infer_jigsaw19.py'.format(code_root)
                    subprocess.call("python {} --do_lower_case                          \
                            --data_dir {}                                               \
                            --bert_model {}                                             \
                            --max_seq_length {}                                         \
                            --output_dir {}                                             \
                            --output_file {}.csv".format(code_file, data_root, bert_model, msl, output_dir, full_name), shell=True)