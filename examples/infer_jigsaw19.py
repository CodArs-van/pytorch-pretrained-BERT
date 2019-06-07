import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss, Softmax
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertForSequenceClassificationContext, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from jigsaw_utils import InputExample, InputFeatures, JigsawRegressionBinaryProcessor, JigsawRegressionProcessor
from jigsaw_utils import processors, output_modes, models, convert_examples_to_features, deserialize_features, serialize_features

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--cls_model",
                        default="default",
                        type=str,
                        help="Classification to use, could be 'default', 'context', 'deepcontext'")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The output submission file name.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--infer_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for inference.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--cache_dir",
                        default="./",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    args = parser.parse_args()

    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))

    model = models[task_name].from_pretrained(args.bert_model,
            cache_dir=cache_dir,
            num_labels=num_labels)

    model.to(device)

    infer_examples = processor.get_test_examples(args.data_dir)
    
    # DEBUG OPTIONS
    # infer_examples = infer_examples[:200]

    infer_features = convert_examples_to_features(
        infer_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(infer_examples))
    logger.info("  Batch size = %d", args.infer_batch_size)
    all_input_ids   = torch.tensor([f.input_ids for f in infer_features],   dtype=torch.long)
    all_input_mask  = torch.tensor([f.input_mask for f in infer_features],  dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in infer_features], dtype=torch.long)
    all_label_ids   = torch.tensor([f.label_id for f in infer_features],    dtype=torch.float)

    infer_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    infer_sampler = SequentialSampler(infer_data)
    infer_dataloader = DataLoader(infer_data, sampler=infer_sampler, batch_size=args.infer_batch_size)

    model.eval()
    nb_infer_steps = 0
    preds = []

    start_index = 0
    for input_ids, input_mask, segment_ids, label_ids in tqdm(infer_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        if task_name == "jigsaw-r-s":
            logits = logits
        elif task_name == "jigsaw-b-s":
            softmax = Softmax(dim=-1)
            logits = softmax(logits)[:, -1].unsqueeze(-1)
        elif task_name == "jigsaw-u-s":
            logits = Softmax(dim=-1)(logits)[:, -1].unsqueeze(-1)
        else:
            raise ValueError("Not supported task")
            
        nb_infer_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                
        start_index += len(input_ids)

    preds = preds[0]
    preds = np.squeeze(preds)
    
    df = pd.DataFrame()
    df['id'] = [example.guid.split('-')[1] for example in infer_examples]
    df['prediction'] = preds
    df.to_csv(os.path.join(args.output_dir, args.output_file), index=False)

if __name__ == "__main__":
    main()
