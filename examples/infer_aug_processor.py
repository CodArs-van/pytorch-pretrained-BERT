from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import shutil
import logging
import argparse
import random
from tqdm import tqdm, trange
import json
import pandas as pd

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_csv(cls, input_file, keeps=None):
        """Read a comma separated value file"""
        df = pd.read_csv(input_file)

        if not keeps:
          columns = [df[col] for col in df.columns]
        else:
          columns = [df[col] for col in df.columns[keeps]]

        lines = []
        for row in zip(*columns):
          line = [ele for ele in row]
          lines.append(line)
        return lines


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        """
        self.guid = guid
        self.text_a = text_a
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, init_ids, input_ids, input_mask, segment_ids, masked_lm_labels):
        self.init_ids = init_ids
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.masked_lm_labels = masked_lm_labels


class JigsawProcessor(DataProcessor):
    """Processor for the Jigsaw data set (Kaggle version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv"), keeps=[0, 1, 2]), "train")

    def get_eval_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "eval.csv"), keeps=[0, 1, 2]), "eval")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def _create_examples(self, lines, set_type):
        raise NotImplementedError()

    def pad_examples(self, set_type, examples, stride, use_str):
        """Pad examples to fix multi-gpu issue"""
        length = len(examples)
        if length % stride != 0:
            n_pad = stride - (length % stride)
            logger.info("  Num pad = %d", n_pad)
            for i in range(length, length + n_pad):
                guid = "%s-%s-pad" % (set_type, i)
                text_a = "This is just for padding"
                label = '0' if use_str else 0.0
                examples.append(
                    InputExample(guid=guid, text_a=text_a, label=label))
        return examples

class JigsawRegressionProcessor(JigsawProcessor):
    """Processor for the Jigsaw data set (Kaggle version)."""

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for line in lines:
            guid = "{}-{}".format(set_type, line[0])
            text_a, label = (line[1], "0") if set_type == "test" else (line[2], "1" if float(line[1]) >= 0.5 else "0")
            examples.append(InputExample(guid, text_a, label))
        return examples


class JigsawRegressionBinaryProcessor(JigsawRegressionProcessor):
    """Processor for the Jigsaw data set (Kaggle version)."""

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    # ----
    masked_lm_prob = 0.15
    max_predictions_per_seq = 20
    rng = random.Random(12345)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        segment_id = label_map[example.label]
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        # 由于是CMLM，所以需要用标签
        tokens = []
        segment_ids = []
        # is [CLS]和[SEP] needed ？
        tokens.append("[CLS]")
        segment_ids.append(segment_id)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(segment_id)
        tokens.append("[SEP]")
        segment_ids.append(segment_id)
        masked_lm_labels = [-1] * max_seq_length

        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            cand_indexes.append(i)

        rng.shuffle(cand_indexes)
        len_cand = len(cand_indexes)

        output_tokens = list(tokens)

        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(tokens) * masked_lm_prob))))

        masked_lms_pos = []
        covered_indexes = set()
        for index in cand_indexes:
            if len(masked_lms_pos) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = tokens[cand_indexes[rng.randint(0, len_cand - 1)]]

            masked_lm_labels[index] = tokenizer.convert_tokens_to_ids([tokens[index]])[0]
            output_tokens[index] = masked_token
            masked_lms_pos.append(index)

        init_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(output_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            init_ids.append(0)
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)  # ?segment_id

        assert len(init_ids) == max_seq_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("init_ids: %s" % " ".join([str(x) for x in init_ids]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("masked_lm_labels: %s" % " ".join([str(x) for x in masked_lm_labels]))

        features.append(
            InputFeatures(init_ids=init_ids,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          masked_lm_labels=masked_lm_labels))
    return features

def rev_wordpiece(str):
    #print(str)
    if len(str) > 1:
        for i in range(len(str)-1, 0, -1):
            if str[i] == '[PAD]':
                str.remove(str[i])
            elif len(str[i]) > 1 and str[i][0]=='#' and str[i][1]=='#':
                str[i-1] += str[i][2:]
                str.remove(str[i])
    return " ".join(str[1:-1])

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="datasets", type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default="aug_data", type=str,
                        help="The output dir for augmented dataset")
    parser.add_argument("--output_name", default="train_aug.csv", type=str,
                        help="The name of the output csv file")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="The path of pretrained bert model.")
    parser.add_argument("--task_name",default="subj",type=str,
                        help="The name of the task to train.")
    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case", default=False, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--infer_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    print(args)
    run_aug(args, save_every_epoch=False)

def run_aug(args, save_every_epoch=False):
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    processors = {
        # you can your processor here
        "jigsaw-b-s": JigsawRegressionBinaryProcessor,
    }

    task_name = args.task_name
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    processor = processors[task_name]()
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_infer_steps = None
    train_examples = processor.get_train_examples(args.data_dir)

    num_infer_steps = int(len(train_examples) / args.infer_batch_size)

    # Prepare model
    model = BertForMaskedLM.from_pretrained(args.bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)

    model.to(device)

    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer)

    logger.info("***** Running inferencing *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.infer_batch_size)
    logger.info("  Num steps = %d", num_infer_steps)
    all_init_ids = torch.tensor([f.init_ids for f in train_features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_masked_lm_labels = torch.tensor([f.masked_lm_labels for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_init_ids, all_input_ids, all_input_mask, all_segment_ids, all_masked_lm_labels)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.infer_batch_size)

    MASK_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

    model.eval()
    aug_comment = []
    train_data_path = os.path.join(args.data_dir, "train.csv")
    df_train = pd.read_csv(train_data_path)
    df = pd.DataFrame()
    df['id'] = df_train['id']
    df['target'] = df_train['target']
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        init_ids, _, input_mask, segment_ids, _ = batch
        input_lens = [sum(mask).item() for mask in input_mask]
        masked_idx = np.squeeze([np.random.randint(0, l, max(l // 7,2)) for l in input_lens])
        for ids, idx in zip(init_ids, masked_idx):
            ids[idx] = MASK_id
        predictions = model(init_ids, segment_ids, input_mask)
        for ids, idx, preds, seg in zip(init_ids, masked_idx, predictions, segment_ids):
            pred = torch.argsort(preds)[:, -2][idx]
            ids[idx] = pred
            new_str = tokenizer.convert_ids_to_tokens(ids.cpu().numpy())
            new_str = rev_wordpiece(new_str)
            aug_comment.append(new_str)
        torch.cuda.empty_cache()
    df['comment_text'] = aug_comment
    df['comment_text_ori'] = df_train['comment_text']
    df.to_csv(os.path.join(args.output_dir, args.output_name), index=False)

if __name__ == "__main__":
    main()