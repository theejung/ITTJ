import nltk
import os,sys
import numpy as np

def load_embedding(emb_path, emb_filename, vocab=None, emb_size=False ):
  import gensim
  emb_dict = gensim.models.KeyedVectors.load_word2vec_format(emb_path+emb_filename, binary=True)
  print 'Embedding Loaded..',emb_path+emb_filename

  if vocab is None:
    return emb_dict

  emb_size_data = len(emb_dict['the'])
  emb = list()
  cnt = 0
  for word in vocab:
    if word in emb_dict.vocab:
        emb.append(emb_dict[word])
        cnt += 1
    else:
        emb.append(np.random.uniform(-1,1,emb_size_data))
  emb = np.array(emb)
  if emb_size:
    assert emb_size == emb.shape[1]
  print "GloVe Loading complete!", emb.shape, cnt, 'exists in glove'
  return emb


def load_glove(emb_path, emb_filename, vocab, emb_size=False):
    emb_dict = dict()
    emb_size_data = 0

    file = open(os.path.join(emb_path, emb_filename), 'r')
    for line in file.readlines():
        row = line.strip().split(" ")
        if row[0] in vocab:
            emb_dict[row[0]] = row[1:]

    emb_size_data = len(row[1:])
    emb = list()

    for word in vocab:
        if word in emb_dict.keys():
            emb.append(emb_dict[word])
        else:
            emb.append(np.random.uniform(-1,1,emb_size_data))


    emb = np.array(emb)

    if emb_size:
        assert emb_size == emb.shape[1]

    print "GloVe Loading complete!", emb.shape
    return emb


def lowerr(arr):
    out = list()
    lenn = list()
    for i in arr:
      tks = i.lower().split() # nltk.word_tokenize(i.lower().decode('utf-8'))
      out.append(' '.join(tks))
      lenn.append(len(tks))
    return out, lenn

def confusion_matrix(expected, predicted, n_classes):
    m = [[0] * n_classes for i in range(n_classes)]
    for pred, exp in zip(predicted, expected):
        m[pred][exp] += 1
    return m

def prec_rec(array):
    precision_out = list()
    recall_out = list()
    fscore_out = list()
    for i in range(len(array)):
        recall_out.append(round(array[i][i] * 1.0 / sum(array[i]) * 100, 2))
        precision_tmp = [x[i] for x in array]

        if sum(precision_tmp) > 0:
            precision_out.append(round(array[i][i] * 1.0 / sum(precision_tmp) * 100, 2))
        else:
            precision_out.append(0)

        precision_, recall_ = precision_out[-1], recall_out[-1]
        fscore_out.append(2 * (precision_ * recall_) / ((precision_ + recall_) * 100))

    return precision_out, recall_out, fscore_out

#NOTE: res['Y], res['length']
class PaddingDatawithTarget():
    def __init__(self, df):
        self.df = df  # pd.DataFrame({'X':X,'Y':Y, 'length':length})
        self.size = len(self.df)
        self.cursor = 0
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor + n > self.size:
            #import pdb; pdb.set_trace()
            self.epochs += 1
            self.shuffle()

        res = self.df.ix[self.cursor:self.cursor + n - 1]
        self.cursor += n

        new_x = []

        for i in res['X']:
          new_x.append(i)

        new_x = np.array(new_x, dtype=np.int32)

        return new_x, np.array(res['Y'], dtype=np.int32), np.array(res['length'], dtype=np.int32)

    def next_batch_text(self, n):
        if self.cursor + n > self.size:
            self.epochs += 1
            self.shuffle()

        res = self.df.ix[self.cursor:self.cursor + n - 1]
        self.cursor += n
        return str(res['X'].values[0]), str(res['Y'].values[0]), None




