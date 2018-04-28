#! /usr/bin/python

import sys, numpy as np
from itertools import izip
from keras.preprocessing.sequence import pad_sequences


np.random.seed(1337)  # for reproducibility


def read(fn, word2index={}, startIndex=0):
    # word2index = {}
    x = []
    y = []
    flag = True
    mxlen = 0
    curIndex = startIndex
    if startIndex is None:
        maxIndex = max(word2index.values()) + 1
    for line in open(fn):
        line = line.strip()
        # print "|%s|"%line
        try:
            sentence, label = line.split("\t")
        except ValueError:
            label = line.split("\t")[0]
            sentence = ""
        lx = []
        y.append(int(label))
        for word in sentence.split():
            if word not in word2index:
                if startIndex is not None:
                    word2index[word] = curIndex
                    curIndex += 1
                else:  # this is for dev and test
                    word2index[word] = maxIndex
            index = word2index[word]
            lx.append(index)
        mxlen = max(mxlen,len(lx))
        x.append(lx)
    print mxlen
    return np.array(x), np.array(y), word2index

def getArgLex(fn1,fn2,fn3):
    arglex = []
    vocab = {}
    reverse_vocab = {}
    vocab_index = 1
    arglex_map = {}
    for line in open('real-arg-lexicons.txt','r'):
        arglex.append(line.strip())
        vocab[line.strip()] = vocab_index
        reverse_vocab[vocab_index] = line.strip()       # generate reverse vocab as well
        vocab_index += 1
    vocab['UNK'] = len(vocab) + 1
    reverse_vocab[len(vocab)] = 'UNK'
    with open("real-arg-lexicons.txt") as textfile1, open("embeddings.txt") as textfile2: 
        for x, y in izip(textfile1, textfile2):
            x = x.strip()
            y = y.strip()
            arglex_map[x] = [float(l) for l in y.split(',')]
    W = get_embedding_weights(vocab,arglex_map)
    X_train = []
    for line in open(fn1):
        a = []
        for i in range(len(arglex)):
            if arglex[i] in line:
                a.append(vocab.get(arglex[i], vocab['UNK']))
        X_train.append(a)
    X_dev = []
    for line in open(fn2):
        a = []
        for i in range(len(arglex)):
            if arglex[i] in line:
                a.append(vocab.get(arglex[i], vocab['UNK']))
        X_dev.append(a)
    X_test = []
    for line in open(fn3):
        a = []
        for i in range(len(arglex)):
            if arglex[i] in line:
                a.append(vocab.get(arglex[i], vocab['UNK']))
        X_test.append(a)

    MAX_SEQUENCE_LENGTH = max(map(lambda x:len(x), X_train))
    MAX_SEQUENCE_LENGTH = max(MAX_SEQUENCE_LENGTH,max(map(lambda x:len(x), X_dev)))
    MAX_SEQUENCE_LENGTH = max(MAX_SEQUENCE_LENGTH,max(map(lambda x:len(x), X_test)))
    print "max seq length is %d"%(MAX_SEQUENCE_LENGTH)
    train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
    dev = pad_sequences(X_dev, maxlen=MAX_SEQUENCE_LENGTH)
    test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
    return train,dev,test , W , vocab

def get_embedding_weights(vocab,arglex_map):
    embedding = np.zeros((len(vocab) + 1, 300))
    n = 0
    for k, v in vocab.iteritems():
        try:
            embedding[v] = arglex_map[k]
        except Exception, e:
            n += 1
            pass
    print "%d embedding missed"%n
    return embedding

if __name__ == "__main__":
    word2index = {}
    train_x, train_y, word2index = read(sys.argv[1], word2index)
    dev_x, dev_y, _ = read(sys.argv[2], word2index, None)
    test_x, test_y, _ = read(sys.argv[3], word2index, None)
    print train_x.shape
    print train_x[100]
