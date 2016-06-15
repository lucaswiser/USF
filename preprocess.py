import nltk
from collections import Counter
import os
import pickle
neg_dir = 'aclImdb/train/neg/'
pos_dir = 'aclImdb/train/pos/'




def get_toks(datafile):
    tok_sents = []   
    with open(datafile) as f:
        sents = f.read().split("<br /><br />")
        for s in sents:
            toks = nltk.tokenize.word_tokenize(s.lower())
            tok_sents.append(toks)
    return tok_sents

def get_chars(datafile):
    char_sents = []
    with open(datafile) as f:
        sents = f.read().split("<br /><br />")
        chars = [c for _sent in sents for c in _sent]
        char_sents.append(chars)
    return char_sents

def get_char_toks(datafile):
    tok_sents = []
    with open(datafile) as f:
        sents = f.read().split("<br /><br />")
        for s in sents:
            toks = s.split() #Just splitting on whitespace here
            tok_sents.append(toks)
    return toks

def extract_word_counts(data_dir, counter):
    for i,f in enumerate(os.listdir(data_dir)):
        counter.update([tok for sent in get_toks(data_dir + f) for tok in sent])
        if i % 100 == 0:
            print("%s pages from %s processed"%(i,data_dir))

def extract_char_counts(data_dir, counter):
    for i,f in enumerate(os.listdir(data_dir)):
        counter.update([c for sent in get_chars(data_dir + f) for c in sent])
        if i % 100 == 0:
            print("%s pages from %s processed"%(i,data_dir))

def main():
    tok_counter = Counter()
    char_counter = Counter()
    print("Extracting tokens")
    extract_word_counts(neg_dir, tok_counter)
    extract_word_counts(pos_dir, tok_counter)
    print("Extracting characters")
    extract_char_counts(neg_dir, char_counter)
    extract_char_counts(pos_dir, char_counter)
    print("10 most common words:")
    print(tok_counter.most_common(10))
    print("10 least common words:")
    print(tok_counter.most_common()[-10:][::-1])
    print("10 most common characters")
    print(char_counter.most_common(10))
    print("10 least common characters")
    print(char_counter.most_common()[-10:][::-1])
    
    spec_word_symbols = ["*PAD*","*UNK*"]
    tok_map = {}
    tok_inv_map = {}
    for i, tok in enumerate(spec_word_symbols):
        tok_map[tok] = i
        tok_inv_map[i] = tok
    word_index = 2
    for tok, count in tok_counter.items():
        if count > 1:
            tok_map[tok] = word_index
            tok_inv_map[word_index] = tok
            word_index += 1
    spec_char_symbols = ["*PAD*","*UNK*","*START*","*END*"]
    char_map = {}
    char_inv_map = {}
    for i, c in enumerate(spec_char_symbols):
        char_map[c] = i
        char_inv_map[i] = c
    char_index = 2
    for c, count in char_counter.items():
        if count > 1:
            char_map[c] = char_index
            char_inv_map[char_index] = c
            char_index += 1

    print("Saving mappings")
    with open('tok_map.pkl','wb') as f:
        pickle.dump(tok_map, f)
    with open('tok_inv_map.pkl','wb') as f:
        pickle.dump(tok_inv_map, f)
    with open('char_map.pkl','wb') as f:
        pickle.dump(char_map, f)
    with open('char_inv_map.pkl','wb') as f:
        pickle.dump(char_inv_map, f)





if __name__ == '__main__':
    main()

