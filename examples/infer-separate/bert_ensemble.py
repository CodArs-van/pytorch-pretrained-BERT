import logging
import pandas as pd
import subprocess

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logger = logging.getLogger(__name__)

def PerformEnsemble(lists, output):
    dfo = pd.DataFrame()
    weight_sum = 0
    for index, (weight, file) in enumerate(lists):
        weight_sum += weight
        dfi = pd.read_csv(file)
        if index == 0:
            dfo['id'] = dfi['id']
            dfo['prediction'] = weight * dfi['prediction'].astype(float)
        else:
            dfo['prediction'] += weight * dfi['prediction'].astype(float)
    dfo['prediction'] = dfo['prediction'] / weight_sum
    print(dfo['prediction'][4])
    dfo.to_csv(output, index=False)

if __name__ == "__main__":
    script = "C:\\Users\\xiaguo\\Developer\\PyTorch\\CodArs-pytorch-pretrained-BERT\\examples\\infer-separate\\bert_allinone.py"
    data_dir = "C:\\Users\\xiaguo\\Developer\\Philly\\Jigsaw2019"
    bs = 32
    bert_models = [
        "C:\\Users\\xiaguo\\Developer\\Philly\\Jigsaw2019\\outs\\Ensemble\\Candidates\\js_base_jigsaw-b-s_msl360_bs512_lr3e5_n4_sd42-epoch1",
        "C:\\Users\\xiaguo\\Developer\\Philly\\Jigsaw2019\\outs\\Ensemble\\Candidates\\js_base_jigsaw-b-s_msl360_bs512_lr3e5_n4_sd42-epoch1",
    ]

    lists = []
    for i, bert_model in enumerate(bert_models):
        output_path = "{}.csv".format(i)
        command = "python {} --data_dir {} --bert_model {} --output_path {} --infer_batch_size {}".format(
            script, data_dir, bert_model, output_path, bs)
        logger.info(command)
        subprocess.call(command, shell=True)
        lists.append((1.0, output_path))

    logger.info("Perform ensemble")
    PerformEnsemble(lists, "submission.csv")