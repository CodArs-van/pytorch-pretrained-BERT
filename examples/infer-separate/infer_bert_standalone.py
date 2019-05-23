from bert_allinone import inference
from pathlib import Path

if __name__ == "__main__":
    home = str(Path.home())
    data_dir = "{}\\Developer\\PyTorch\\BERT\\Kaggle-Jigsaw".format(home)
    bert_model = "{}\\Developer\\Kaggle\\Jigsaw\\train_model\\js-base-msl384-bs32-lr2e5-nepoch2".format(home)
    task_name = "jigsaw-r"
    output_path = "{}\\Developer\\Kaggle\\Jigsaw\\infer_output\\js-base-msl384-bs32-lr2e5-nepoch2-cached3.csv".format(home)
    feature_cache_dir = "{}\\Developer\\Kaggle\\Jigsaw\\feature_cache".format(home)
    max_seq_length = 384
    do_lower_case = True
    infer_batch_size = 8
    cls_model = 'cls-classical'
    inference(data_dir, bert_model, task_name, output_path, 
        feature_cache_dir, max_seq_length, do_lower_case, cls_model, infer_batch_size)
