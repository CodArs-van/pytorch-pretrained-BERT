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

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


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

    def pad_examples(self, set_type, examples, stride):
        """Pad examples to fix multi-gpu issue"""
        return examples

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
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
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
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
            text_a, label = (line[1], "0.5") if set_type == "test" else (line[2], line[1])
            examples.append(InputExample(guid, text_a, None, label))
        return examples


class JigsawRegressionBinaryProcessor(JigsawRegressionProcessor):
    """Processor for the Jigsaw data set (Kaggle version)."""

    def get_labels(self):
        """See base class."""
        return [None, None]


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def deserialize_features(feature_cache_path):
    with open(feature_cache_path, "rb") as f:
        return pickle.load(f)


def serialize_features(feature_cache_path, features):
    with open(feature_cache_path, "wb") as f:
        pickle.dump(features, f)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
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
        "--cls_model",
        default="default",
        type=str,
        help="classification model, can be 'default', 'context', 'deep'")
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

    processors = {
        "jigsaw-r": JigsawRegressionProcessor,
        "jigsaw-b": JigsawRegressionBinaryProcessor,
    }

    output_modes = {
        "jigsaw-r": "regression",
        "jigsaw-b": "regression"
    }

    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(
        format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt = '%m/%d/%Y %H:%M:%S',
        level = logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    train_examples = None
    train_examples = processor.get_train_examples(args.data_dir)
    train_examples = processor.pad_examples('train', train_examples, args.train_batch_size, use_str=True)

    num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size) * args.num_train_epochs
    cache_dir = args.cache_dir
    if args.cls_model == "default":
        model = BertForSequenceClassification.from_pretrained(args.bert_model,
                cache_dir=cache_dir,
                num_labels=num_labels)
    elif args.cls_model == "context":
        model = BertForSequenceClassificationContext.from_pretrained(args.bert_model,
                cache_dir=cache_dir,
                num_labels=num_labels)
    else:
        raise NotImplementedError()

    model.to(device)
    model = torch.nn.DataParallel(model)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
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
    feature_cache_path = os.path.join(args.feature_cache_dir, feature_cache_file)

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

    all_input_ids   = torch.tensor([f.input_ids for f in train_features],   dtype=torch.long)
    all_input_mask  = torch.tensor([f.input_mask for f in train_features],  dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids   = torch.tensor([f.label_id for f in train_features],    dtype=torch.float)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    model.train()

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps, nb_tr_steps_save, nb_tr_steps_save_index = 0, 0, 0, 0
        nb_tr_steps_total = len(train_dataloader)
        nb_tr_steps_5percent = nb_tr_steps_total // 20
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            logits = model(input_ids, segment_ids, input_mask, labels=None)

            if task_name == "jigsaw-r":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))
            elif task_name == "jigsaw-b":
                logsoftmax = LogSoftmax(dim=-1)
                label_ids = label_ids.unsqueeze(1)
                soft_labels = torch.cat([1 - label_ids, label_ids], dim=1)
                loss = torch.sum(-soft_labels * logsoftmax(logits))
            else:
                raise ValueError("task not supported")

            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.

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
            if nb_tr_steps_save >= nb_tr_steps_5percent and epoch > 0:
                nb_tr_steps_save = 0
                # Save a trained model, configuration and tokenizer
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                # If we save using the predefined names, we can load using `from_pretrained`
                subdir = 'epoch_{}-index_{}'.format(epoch, nb_tr_steps_save_index)
                output_dir = os.path.join(args.output_dir, subdir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                output_config_file = os.path.join(output_dir, CONFIG_NAME)

                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(output_dir)

                nb_tr_steps_save_index += 1

if __name__ == "__main__":
    main()
