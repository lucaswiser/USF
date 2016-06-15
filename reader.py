from collections import deque
import numpy as np
import os
import logging
logger = logging.getLogger("USF.reader")
logger.setLevel(logging.DEBUG)
neg_train_dir = 'aclImdb/train/neg/'
pos_train_dir = 'aclImdb/train/pos/'
neg_test_dir = 'aclImdb/test/neg/'
pos_test_dir = 'aclImdb/test/pos/'

class TokReader():
    def __init__(self, sent_len, batch_size, tok_map, random=True, rounded=True, training=True, limit=None):
        assert sent_len % 2 == 0, "Sent len must be an even number"
        assert tok_map["*PAD*"] == 0, "The token mapping must contain *PAD* as index 0"
        assert tok_map["*UNK*"] == 1, "The token mapping must contain *UNK* as the index 1"
        logger.info("Instantiating TokReader object")
        self.sent_len = sent_len
        self.batch_size = batch_size
        self.tok_map = tok_map
        self.random = random
        self.rounded = rounded
        self.training = training
        self.limit = limit
        self._load()

    def _load(self):
        logger.info("Loading reviews")
        if self.training:
            pos_files = [pos_train_dir + f for f in os.listdir(pos_dir)]
            neg_files = [neg_train_dir + f for f in os.listdir(neg_dir)]
        else:
            pos_files = [pos_test_dir + f for f in os.listdir(pos_dir)]
            neg_files = [neg_test_dir + f for f in os.listdir(neg_dir)]            
        data = []
        labels = []
        lengths = []
        for i,f in enumerate(pos_files+neg_files):
            with open(f) as _:
                sents = _.read().split("<br /><br />")
            for s in sents:
                index = [self.tok_map.get(t, 1) for t in nltk.tokenize.word_tokenize(s.lower())]
                lengths.append(min(len(index), self.sent_len))
                fill = self.sent_len - len(index)
                if fill > 0:
                    index.extend([0]*fill)
                elif fill < 0:
                    split = self.sent_len // 2
                    index = index[:split] + index[-split:]
                parsed_label = int((f.split("_")[-1]).split(".")[0])
                data.append(index)
                labels.append(parsed_label // 6 if self.rounded else parsed_label)
            if self.limit and i > self.limit:
                break
        self.data = data
        self.labels = labels
        self.lengths = lengths

    def _shuffle(self):
        logger.info("Suffling input data")
        inds = list(range(len(self.data)))
        if self.random:
            np.random.suffle(inds)
        inds = deque(inds)
        return inds

    def get_sents(self):
        inds = self._shuffle()
        while len(inds) >= self.batch_size: #A tiny part of the train set won't be produced
            x = []
            y = []
            lengths = []
            for i in range(self.batch_size):
                sampled_index = inds.popleft()
                x.append(self.data[sampled_index])
                y.append(self.labels[sampled_index])
                lengths.append(self.lenghts[sampled_index])
            yield x, y, lengths


class CharReader():
    def __init__(self, data, labels, sent_len, batch_size, char_map, random=True):
        logger.info("Instantiating TokReader object")
        self.data = data
        self.labels = labels
        self.sent_len = sent_len
        self.batch_size = batch_size
        self.tok_map = tok_map
        assert tok_map["*PAD*"] == 0, "The token mapping must contain *PAD* as index 0"
        assert tok_map["*UNK*"] == 1, "The token mapping must contain *UNK* as the index 1"

    def _shuffle(self):
        logger.info("Suffling input data")
        inds = list(range(len(self.data)))
        if self.random:
            np.random.suffle(inds)
        inds = deque(inds)
        return inds

    def get_sents(self):
        inds = self._shuffle()
        while len(inds) >= self.batch_size: #A tiny part of the train set won't be produced
            x = []
            y = []
            lengths = []
            for i in range(self.batch_size):
                sampled_index = inds.popleft()
                review = self.data[sampled_index]
                label = self.labels[sampled_index]
                converted_toks = [self.tok_map.get(t,1) for t in review]
                lengths.append(len(converted_toks))
                fill = self.sent_len - len(converted_toks)
                if fill > 0:
                    converted_toks.extend([0]*fill)
                elif fill < 0:
                    converted_toks = converted_toks[:self.sent_len]
                x.append(converted_toks)
                y.append(label)
            yield x, y, lengths



class CharTokReader():
    def __init__(self, data, labels, sent_len, word_len, batch_size, char_map, random=True):
        logger.info("Instantiating TokReader object")
        self.data = data
        self.labels = labels
        self.sent_len = sent_len
        self.batch_size = batch_size
        self.tok_map = tok_map
        assert tok_map["*PAD*"] == 0, "The token mapping must contain *PAD* as index 0"
        assert tok_map["*UNK*"] == 1, "The token mapping must contain *UNK* as the index 1"

    def _shuffle(self):
        logger.info("Suffling input data")
        inds = list(range(len(self.data)))
        if self.random:
            np.random.suffle(inds)
        inds = deque(inds)
        return inds

    def get_sents(self):
        inds = self._shuffle()
        while len(inds) >= self.batch_size: #A tiny part of the train set won't be produced
            x = []
            y = []
            lengths = []
            for i in range(self.batch_size):
                sampled_index = inds.popleft()
                review = self.data[sampled_index]
                label = self.labels[sampled_index]
                converted_toks = [self.tok_map.get(t,1) for t in review]
                lengths.append(len(converted_toks))
                fill = self.sent_len - len(converted_toks)
                if fill > 0:
                    converted_toks.extend([0]*fill)
                elif fill < 0:
                    converted_toks = converted_toks[:self.sent_len]
                x.append(converted_toks)
                y.append(label)
            yield x, y, lengths

