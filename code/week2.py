import os, sys
import pandas as pd
import numpy as np
import zipfile

path = '../data/'
train_zipnm = 'train.tsv.zip'

# road train data into pandas DataFrame
with zipfile.ZipFile(os.path.abspath(path + train_zipnm)) as z:
    with z.open(z.namelist()[0]) as f:
        train = pd.read_csv(f, header=0, delimiter='\t')

print 'Load train data done!'

# tokenize! delete punctuation marks and change upper letter to lower one
#from nltk.tokenize import RegexpTokenizer
#import nltk
#
#def token(array):
#    out = list()
#    tokenizer = nltk.word_tokenize(r'\w+')
#    for i in range(len(array)):
#        out.append(tokenizer(array[i].lower()))
#
#    return out
#

#train_sentences = token(train.Phrase)
#print 'Number of Train', len(train_sentences)
#print train_sentences[:10]

# RNN
# Data Preparation
# -------- Version 1: RNN with BoW (n= 1000) -------
# Top 'n' Freq from 156060 Train sets
#flat_sentences = [x for y in train_sentences for x in y]
#n = 20000  # We can change it if we want!
#print 'Number of Words in Train', len(flat_sentences)
#print flat_sentences[:5]
#import collections
##
#freq_dict = collections.Counter(flat_sentences)
#print 'Top 10 freq words and freq', freq_dict.most_common(10)
#print '1st, 2nd, 3rd freq Words', freq_dict.most_common(100)[0][0], freq_dict.most_common(100)[1][0], freq_dict.most_common(100)[2][0]
##
#freq_dict_n = freq_dict.most_common(n)
#dict = dict()
##
#for i in range(n):
#    dict[freq_dict_n[i][0]] = i + 1
##
#dict['PAD'] = 0
#dict['UNK'] = n + 1
#
#print 'Check whether dict is correct', dict['the'], dict['a'], dict['best'], dict['terms'], dict['UNK']
#print 'Average length of Sentences', sum([len(i) for i in train_sentences]) / (len(train_sentences) * 1.0)
#print 'Percentage of len(Sentence) > 10', sum([len(i) > 10 for i in train_sentences]) / (len(train_sentences) * 1.0)
#
pad_size = 10  # based on average length of sentences: 6.89
#
#
#def sentence_padding(arr):
#    out = list()
#    lenn = list()
#    for phrase in arr:
#        phrase_num = list()
#        len_phrase = len(phrase)
#        for word in phrase:
#            if word in dict.keys():
#                phrase_num.append(dict[word])
#            else:
#                phrase_num.append(dict['UNK'])
#
#        if len(phrase) >= 10:
#            out.append(phrase_num[:10])
#            lenn.append(10)
#        else:
#            out.append(phrase_num + [dict['PAD']] * (10 - len(phrase)))
#            lenn.append(len_phrase)
#
#    return out, lenn
#
#
#X, length = sentence_padding(train_sentences)
#Y = train.Sentiment
#
#print 'First 5 sentences result', X[:5]
#print 'First 5 sentences length', length[:5]
#
## For batch split!
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
#
#
## Train vs Test
#df = pd.DataFrame({'X': X, 'Y': Y, 'length': length})
#df = df.sample(frac=1, random_state = 63).reset_index(drop=True)
#test_len = np.floor(len(df) * 0.1)
#test, train = df.ix[:test_len - 1], df.ix[test_len:]
##print sum(train['Y']), sum(test['Y'])
#
## Data batch Test!
#data = PaddingDatawithTarget(train)
#d = data.next_batch(3)
#
#print 'Length check', len(df['X']), sum([len(i) for i in df['X']])
#print d
#

# -------- Version 2: RNN with pretrained Glove (dim = 300) -------
# vocab based on GloVe and Input!
# raw input -> let's make it lower!
def lowerr(arr):
    out = list()
    lenn = list()
    for i in arr:
        out.append(i.lower())
        lenn.append(len(i))

    return out, lenn

train_sentence, length = lowerr(train.Phrase)
print train_sentence[:5]

# Top n vocab only!
#n = 20000  # We can change it if we want!
flat_sentences = [z for y in [x.strip().split(' ') for x in train_sentence] for z in y]
print 'Number of Words in Train', len(flat_sentences)
print flat_sentences[:100]

import collections
#
freq_dict = collections.Counter(flat_sentences)
print 'Top 10 freq words and freq', freq_dict.most_common(10)
print '1st, 2nd, 3rd freq Words', freq_dict.most_common(100)[0][0], freq_dict.most_common(100)[1][0], freq_dict.most_common(100)[2][0]

n = len(freq_dict.items())
#freq_dict_n = freq_dict.most_common(n)
#vocab_train = list()
#
#for i in range(n):
#    vocab_train.append(freq_dict_n[i][0])
#
print "vocabulary from Train Done!", len(freq_dict.keys())

# Load GloVe
"""

>>> from gensim.models.keyedvectors import KeyedVectors
>>> model = KeyedVectors.load_word2vec_format('/tmp/vectors.bin', binary=True)  # C binary format


for v in freq_dict.keys():
    if v in model:
        mode[v]
    else:
        random.random(300)

"""
filename = 'glove.6B.300d.txt'

def load_glove(filename):
    vocab = []
    embed = []
    file = open(os.path.abspath(filename), 'r')
    for line in file.readlines():
        row = line.strip().split(" ")
        if row[0] in freq_dict.keys():
            vocab.append(row[0])
            embed.append(row[1:])

    print "GloVe Loading complete!"
    return vocab, embed

vocab, embed = load_glove(path+'glove.6B/'+filename)
vocab_size = len(vocab)
embedding_dim = len(embed[0])
embedding = np.asarray(embed)

print "GloVe Vocab done! vocab_size :", vocab_size
# Top 'n' Freq from 156060 Train sets
#flat_sentences = [x for y in train_sentences for x in y]



from tensorflow.contrib import learn
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length = pad_size)
pretrain = vocab_processor.fit(vocab)
X = list(vocab_processor.fit_transform(train_sentence))
print 'X transformation Done!'
print X[:3], length[:3]
print len(X), len(length), len(train.Phrase), len(train.Sentiment)

## Train vs Test
df = pd.DataFrame({'X': X, 'Y': train.Sentiment, 'length': length})
df = df.sample(frac=1, random_state = 63).reset_index(drop=True)
test_len = np.floor(len(df) * 0.1)
test, train = df.ix[:test_len - 1], df.ix[test_len:]
print 'Train/Test split Done!'


# Modeling with TensorFlow
import tensorflow as tf

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

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def build_graph(
        vocab_size=vocab_size,
        hidden_size=256,
        batch_size=64,
        num_class=len(np.unique(train.Y))):
    reset_graph()

    # Plaeholders(keep data)
    x = tf.placeholder(dtype=tf.int32, shape=[batch_size, None]) # [B x L]
    y = tf.placeholder(dtype=tf.int32, shape=[batch_size])
    seqlen = tf.placeholder(dtype=tf.int32, shape=[batch_size])
    keep_prob = tf.constant(1.0)

    # Embedding layer (normal nums turn into vector with lenth of hidden_size
    #TODO replace random initialization of "embeddings" with glove embeddings (hidden_size=300)
    #TODO check "trainable" variable for "embeddings" whether to train the embeddings during training or not -> set trainable = False ve True
    #embeddings = tf.get_variable("embedding_matrix", [vocab_size, hidden_size])
    WW = tf.Variable(tf.constant(0.0, shape = [vocab_size, embedding_dim]), trainable=True, name = 'WW')
    embedding_placeholder = tf.placeholder(tf.float32, shape = [vocab_size, embedding_dim])
    WW.assign(embedding_placeholder)
    rnn_inputs = tf.nn.embedding_lookup(WW, x) # [B x S x 1] x [V x H ] -> [B x S x H]

    # RNN: [B x S x H] -> [B x H]
    cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    init_state = tf.get_variable('init_state', [1, hidden_size]) #, initializer=tf.constant_initializer(0.0))
    init_state = tf.tile(init_state, [batch_size, 1])  # init * [batch_size,1]
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=seqlen, initial_state=init_state)

    # Dropouts
    #TODO double check
    #rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob=keep_prob)

    # Last Output! [ B x H]
    #TODO why?
    print final_state
    #print rnn_outputs, rnn_outputs.get_shape() # [B x ? x H]
    #last_run_output = tf.gather_nd(rnn_outputs, tf.stack([tf.range(batch_size), seqlen - 1], axis=1))
    #print last_run_output, last_run_output.get_shape()
    #sys.exit(1)

    # tf.stack([0,1,2,3],[0,1,2,3], axis = 1)  -> [[0,0],[1,1],[2,2],[3,3]]

    # Softmax layer with last output: [B x H] [ H x N ] = [B x N]
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [hidden_size, num_class])
        b = tf.get_variable('b', [num_class])

    logits = tf.matmul(final_state, W) + b
    preds = tf.nn.softmax(logits)
    correct = tf.equal(tf.cast(tf.argmax(preds, 1), tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # precision & recall
    preds_num = tf.cast(tf.argmax(preds, 1), tf.int32)
    # loss([B x N], y[B x N])
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    train_step = tf.train.AdamOptimizer(5e-3).minimize(loss)

    return {
        'x': x,
        'seqlen': seqlen,
        'y': y,
        #'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'preds': preds,
        'accuracy': accuracy,
        'preds_num': preds_num,
        'num_class' : num_class
    }


def train_graph(graph, batch_size=256, num_epochs=10, num_class=5, iterator=PaddingDatawithTarget):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        tr = iterator(train)
        te = iterator(test)

        step, accuracy, loss = 0, 0, 0.
        tr_losses, te_losses = [], []
        te_cm, te_precision, te_recall, te_fmeasure = [],[],[],[]
        current_epoch = 0

        num_batch_train = tr.size / batch_size

        while current_epoch < num_epochs:
            step += 1
            #tr.shuffle()
            batch = tr.next_batch(batch_size)
            # print batch[0], batch[1], batch[2]
            feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2]} #, g['dropout']: 0.7}
            accuracy_,loss_, _ = sess.run([g['accuracy'], g['loss'], g['ts']], feed_dict=feed)
            accuracy += accuracy_
            loss += loss_

            if step % 100 == 0:
                print "[%d/%d] Train Loss %.4f, Accuracy %.4f "%(step, num_batch_train, 1.0*loss/step, 100.0*accuracy/step)

            if tr.epochs > current_epoch:
                current_epoch += 1
                tr_losses.append(accuracy / step)
                step, accuracy = 0, 0

                te_epoch = te.epochs
                while te.epochs == te_epoch:
                    step += 1
                    batch = te.next_batch(batch_size)
                    # print batch[0], batch[1], batch[2]
                    feed = {g['x']:batch[0], g['y']: batch[1], g['seqlen']:batch[2]}
                    accuracy_, preds_num_ = sess.run([g['accuracy'],g['preds_num']], feed_dict=feed)
                    #print preds_num_ , batch[1]
                    #sys.exit(1)
                    rnn_conf = confusion_matrix(preds_num_,batch[1],num_class)
                    #precision_, recall_,fmeasure_= prec_rec(rnn_conf)
                    accuracy += accuracy_
                    if te_cm == []:
                        te_cm = np.array(rnn_conf)
                    else:
                        te_cm += np.array(rnn_conf)

                print te_cm

                precision, recall, fmeasure = prec_rec(te_cm)
                te_losses.append(accuracy / step)
                te_precision.append(precision)
                te_recall.append(recall)
                te_fmeasure.append(fmeasure)
                step, accuracy, loss = 0, 0, 0.
                te_cm = []
                print "\tAccuracy after epoch %d, train acc %.2f test acc %.2f" %(current_epoch, 100.0 * tr_losses[-1], 100.0 * te_losses[-1]), te_precision[-1], te_recall[-1], te_fmeasure[-1]

    return tr_losses, te_losses


# Run!
batch_size = 128 # 128
hidden_size = 256 # 512
num_class = len(np.unique(train.Y))
g = build_graph(batch_size = batch_size, hidden_size = hidden_size, num_class = num_class)
tr_losses, te_losses = train_graph(g, batch_size = batch_size, num_class = num_class)
