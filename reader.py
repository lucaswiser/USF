from collections import deque
import numpy as np
import logging
logger = logging.getLogger("USF.reader")
logger.setLevel(logging.DEBUG)

class TokReader():
    def __init__(self, data, labels, sent_len, batch_size, tok_map, random=True):
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

