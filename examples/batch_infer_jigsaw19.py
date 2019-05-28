import subprocess
import os
import sys
import re

if __name__ == '__main__':
    begin_dir = 'C:\\Users\\xiaguo\\Developer\\Philly\\Jigsaw2019\\kgjs19'
    target_dir = 'js_base_jigsaw-b_msl384_bs64_lr2e5_n3_seed8_clsdefault'
    data_dir = 'C:\\Users\\xiaguo\\Developer\\Philly\\Jigsaw2019'
    output_dir = 'C:\\Users\\xiaguo\\Developer\\PyTorch\\CodArs-pytorch-pretrained-BERT\\.tmp\\jigsaw_output'
    msl = re.search(r'msl(\d+)_', target_dir)[1]
    cls_model = re.search(r'cls(\w+)', target_dir)[1]
    task_name = re.search(r'base_(.+)_msl', target_dir)[1]
    root_dir = os.path.join(begin_dir, target_dir)
    for subdir in [f.name for f in os.scandir(root_dir) if f.is_dir()]:
        if subdir.startswith('epoch_1'):
            print('Skipping...{}'.format(subdir))
            continue
        print('Infer for {}'.format(subdir))
        filename = '{}_{}.csv'.format(target_dir, subdir)
        bert_model = os.path.join(root_dir, subdir)
        subprocess.call('python infer_jigsaw19.py --do_lower_case --data_dir {} --bert_model {} \
            --max_seq_length {} --output_dir {} --output_file {} --task_name {} --cls_model {}  \
            '.format(data_dir, bert_model, msl, output_dir, filename, task_name, cls_model), shell=True)