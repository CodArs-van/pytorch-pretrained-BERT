import csv
import logging
import os
import sys
import pandas as pd
import pickle

from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertForSequenceClassificationContext

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

class JigsawUDAProcessor(JigsawRegressionBinaryProcessor):
    """Processor for the Jigsaw data set (Kaggle version)."""

    def get_unsupervised_examples(self, data_dir):
        return self._create_unsup_examples(
            self._read_csv(os.path.join(data_dir, "kgjs_fren.csv")), "unsup")

    def _create_unsup_examples(self, lines, set_type):
        """Create examples for unsupervised training"""
        examples_ori, examples_bak = [], []
        for line in lines:
            guid = "{}-{}".format(set_type, line[0])
            text_a_ori, text_a_bak, label = (line[2], line[3], "0.5")
            examples_ori.append(InputExample(guid, text_a_ori, None, label))
            examples_bak.append(InputExample(guid, text_a_bak, None, label))
        return examples_ori, examples_bak


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

processors = {
    "jigsaw-r-s": JigsawRegressionProcessor,
    "jigsaw-b-s": JigsawRegressionBinaryProcessor,
    "jigsaw-u-s": JigsawUDAProcessor,
}

output_modes = {
    "jigsaw-r-s": "regression",
    "jigsaw-b-s": "regression",
    "jigsaw-u-s": "regression",
}

models = {
    "jigsaw-r-s": BertForSequenceClassification,
    "jigsaw-b-s": BertForSequenceClassification,
    "jigsaw-u-s": BertForSequenceClassification,
}