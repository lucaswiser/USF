from collections import deque
import numpy as np
import os
import pickle
import logging
import nltk
import sys
from threading import Thread
logger = logging.getLogger("USF.reader")
logger.setLevel(logging.DEBUG)
neg_train_dir = 'aclImdb/train/neg/'
pos_train_dir = 'aclImdb/train/pos/'
neg_test_dir = 'aclImdb/test/neg/'
pos_test_dir = 'aclImdb/test/pos/'


class TokReader():
    def __init__(self, sent_len, batch_size, tok_map, random=True, rounded=True, training=True, limit=None):
        assert sent_len % 2 == 0, "Sent len must be an even number"
        logger.info("Instantiating TokReader object: %s"%("training" if training else "valid"))
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
            pos_files = [pos_train_dir + f for f in os.listdir(pos_train_dir)]
            neg_files = [neg_train_dir + f for f in os.listdir(neg_train_dir)]
        else:
            pos_files = [pos_test_dir + f for f in os.listdir(pos_test_dir)]
            neg_files = [neg_test_dir + f for f in os.listdir(neg_test_dir)] 
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
                    index = index[:self.sent_len]
                parsed_label = int((f.split("_")[-1]).split(".")[0])
                data.append(index)
                labels.append(parsed_label // 6 if self.rounded else parsed_label)
            if self.limit and i > self.limit:
                break
        self.data = data
        self.labels = labels
        self.lengths = lengths

    def _shuffle(self):
        logger.info("Shuffling input data")
        inds = list(range(len(self.data)))
        if self.random:
            np.random.shuffle(inds)
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
                lengths.append(self.lengths[sampled_index])
            yield np.array(x), np.array(y).reshape((-1,1)), np.array(lengths)


class CharReader():
    def __init__(self, sent_len, batch_size, char_map, random=True, rounded=True, training=True, limit=None):
        assert sent_len % 2 == 0, "Sent len must be an even number"
        logger.info("Instantiating CharReader object: %s"%("training" if training else "valid"))
        self.sent_len = sent_len
        self.batch_size = batch_size
        self.char_map = char_map
        self.random = random
        self.rounded = rounded
        self.training = training
        self.limit = limit
        self._load()

    def _load(self):
        logger.info("Loading reviews")
        if self.training:
            pos_files = [pos_train_dir + f for f in os.listdir(pos_train_dir)]
            neg_files = [neg_train_dir + f for f in os.listdir(neg_train_dir)]
        else:
            pos_files = [pos_test_dir + f for f in os.listdir(pos_test_dir)]
            neg_files = [neg_test_dir + f for f in os.listdir(neg_test_dir)] 
        data = []
        labels = []
        lengths = []
        for i,f in enumerate(pos_files+neg_files):
            with open(f) as _:
                sents = _.read().split("<br /><br />")
            for s in sents:
                index = [self.char_map.get(c, 1) for c in s]
                lengths.append(min(len(index), self.sent_len))
                fill = self.sent_len - len(index)
                if fill > 0:
                    index.extend([0]*fill)
                elif fill < 0:
                    index = index[:self.sent_len]
                parsed_label = int((f.split("_")[-1]).split(".")[0])
                data.append(index)
                labels.append(parsed_label // 6 if self.rounded else parsed_label)
            if self.limit and i > self.limit:
                break
        self.data = data
        self.labels = labels
        self.lengths = lengths

    def _shuffle(self):
        logger.info("Shuffling input data")
        inds = list(range(len(self.data)))
        if self.random:
            np.random.shuffle(inds)
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
                lengths.append(self.lengths[sampled_index])
            yield np.array(x), np.array(y).reshape((-1,1)), np.array(lengths)



class CharTokReader():
    def __init__(self, sent_len, word_len, batch_size, char_map, random=True,
                 rounded=True, training=True, limit=None):
        assert sent_len % 2 == 0, "Sent len must be an even number"
        assert word_len % 2 == 0, "Word len must be an even number"
        logger.info("Instantiating CharReader object: %s"%("training" if training else "valid")) 
        self.sent_len = sent_len
        self.word_len = word_len
        self.batch_size = batch_size
        self.char_map = char_map
        self.random = random
        self.rounded = rounded
        self.training = training 
        self.limit = limit
        self._load()

    def _load(self):
        logger.info("Loading reviews")
        if self.training:
            pos_files = [pos_train_dir + f for f in os.listdir(pos_train_dir)]
            neg_files = [neg_train_dir + f for f in os.listdir(neg_train_dir)]
        else:
            pos_files = [pos_test_dir + f for f in os.listdir(pos_test_dir)]
            neg_files = [neg_test_dir + f for f in os.listdir(neg_test_dir)] 
        data = []
        labels = []
        lengths = []
        wordlengths = []
        for i,f in enumerate(pos_files+neg_files):
            with open(f) as _:
                sents = _.read().split("<br /><br />")
            for s in sents:
                toks = s.split()
                index = [[2] + [self.char_map.get(c, 1) for c in _t] + [3] for _t in toks]
                lengths.append(min(len(index), self.sent_len))
                temp = []
                for i, word in enumerate(index):
                    temp.append(min(len(word), self.word_len))
                    wordfill = self.word_len - len(word)
                    if wordfill > 0:
                        word.extend([0]*wordfill)
                    elif wordfill < 0:
                        split = self.word_len // 2
                        index[i] = word[:split] + word[-split:]
                fill = self.sent_len - len(index)
                if fill > 0:
                    index.extend([[0]*self.word_len]*fill)
                    temp.extend([0]*fill)
                elif fill < 0:
                    index = index[:self.sent_len]
                    temp = temp[:self.sent_len]
                parsed_label = int((f.split("_")[-1]).split(".")[0])
                data.append(index)
                labels.append(parsed_label // 6 if self.rounded else parsed_label)
                wordlengths.append(temp)
            if self.limit and i > self.limit:
                break
        self.data = data
        self.labels = labels
        self.lengths = lengths
        self.wordlengths = wordlengths
        print("Loaded %s reviews"%len(data))

    def _shuffle(self):
        logger.info("Shuffling input data")
        inds = list(range(len(self.data)))
        if self.random:
            np.random.shuffle(inds)
        inds = deque(inds)
        return inds

    def get_sents(self):
        inds = self._shuffle()
        while len(inds) >= self.batch_size: #A tiny part of the train set won't be produced
            x = []
            y = []
            lengths = []
            wordlengths = []
            for i in range(self.batch_size):
                sampled_index = inds.popleft()
                x.extend(self.data[sampled_index])
                y.append(self.labels[sampled_index])
                lengths.append(self.lengths[sampled_index])
                wordlengths.extend(self.wordlengths[sampled_index])
            yield np.array(x), np.array(y).reshape((-1,1)), np.array(lengths), np.array(wordlengths)

