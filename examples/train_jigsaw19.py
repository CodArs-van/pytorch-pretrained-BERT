import argparse
import csv
import logging
import os
import pickle
import random
import sys

import pandas as pd
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss, LogSoftmax
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

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument(
        "--bert_model",
        default=None,
        type=str,
        required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument(
        "--cache_dir",
        default=".cache",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
             "Sequences longer than this will be truncated, and sequences shorter \n"
             "than this will be padded.")
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Total batch size for training.")
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
             "E.g., 0.1 = 10%% of training.")
    parser.add_argument(
        "--feature_cache_dir",
        default="./",
        type=str,
        help="cache directory for large training dataset")
    parser.add_argument(
        "--use_feature_cache",
        action='store_true',
        help="Whether to use feature cache")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for initialization")
    args = parser.parse_args()

    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir))

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True)

    train_examples = None
    train_examples = processor.get_train_examples(args.data_dir)
    train_examples = processor.pad_examples(
        'train', train_examples, args.train_batch_size, use_str=True)

    # DEBUG OPTIONS
    # train_examples = train_examples[:200]

    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size) * args.num_train_epochs
    cache_dir = args.cache_dir

    model = models[task_name].from_pretrained(args.bert_model,
                                              cache_dir=cache_dir,
                                              num_labels=num_labels)

    model.to(device)
    model = torch.nn.DataParallel(model)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        warmup=args.warmup_proportion,
        t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if not os.path.exists(args.feature_cache_dir):
        os.makedirs(args.feature_cache_dir)
    feature_cache_file = "task_{}-msl_{}-md_{}-ph_{}.pickle".format(
        task_name, args.max_seq_length, 'uncased', 'train')
    feature_cache_path = os.path.join(
        args.feature_cache_dir, feature_cache_file)

    train_features = []

    if not os.path.exists(feature_cache_path) or not args.use_feature_cache:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode)

        if args.use_feature_cache:
            serialize_features(feature_cache_path, train_features)
    else:
        if args.use_feature_cache:
            train_features = deserialize_features(feature_cache_path)
        else:
            raise ValueError("This is an unreachable path")

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    all_input_ids = torch.tensor(
        [f.input_ids for f in train_features],   dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in train_features],  dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in train_features],    dtype=torch.float)

    train_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    model.train()

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps, nb_tr_steps_save, nb_tr_steps_save_index = 0, 0, 0, 0
        nb_tr_steps_total = len(train_dataloader)
        nb_tr_steps_4percent = nb_tr_steps_total // 25
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            logits = model(input_ids, segment_ids, input_mask, labels=None)

            if task_name == "jigsaw-r-s":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))
            elif task_name == "jigsaw-b-s":
                logsoftmax = LogSoftmax(dim=-1)
                label_ids = label_ids.unsqueeze(1)
                soft_labels = torch.cat([1 - label_ids, label_ids], dim=1)
                loss = torch.sum(-soft_labels * logsoftmax(logits))
            else:
                raise ValueError("task not supported")

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 1000 == 1:
                logger.info("Train Loss: {}".format(tr_loss / global_step))

            nb_tr_steps_save += 1
            if nb_tr_steps_save >= nb_tr_steps_4percent and epoch > 0:
                nb_tr_steps_save = 0
                # Save a trained model, configuration and tokenizer
                model_to_save = model.module if hasattr(
                    model, 'module') else model  # Only save the model it-self
                # If we save using the predefined names, we can load using `from_pretrained`
                subdir = 'e{}-idx{}'.format(epoch, nb_tr_steps_save_index)
                output_dir = os.path.join(args.output_dir, subdir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                output_config_file = os.path.join(output_dir, CONFIG_NAME)

                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(output_dir)

                nb_tr_steps_save_index += 1

        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        epoch_dir = os.path.join(args.output_dir, 'epoch{}'.format(epoch))
        output_model_file = os.path.join(epoch_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(epoch_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(epoch_dir)

    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)


if __name__ == "__main__":
    main()
