import os, sys, argparse
import numpy as np
import tensorflow as tf

from utils import confusion_matrix, prec_rec, PaddingDatawithTarget, load_glove
from data_loader import DataLoader



class SentiRNN():
    def __init__(self,args):
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.emb_size = args.emb_size
        self.emb_trainable = args.emb_trainable
        self.load_glove = args.load_glove
        self.num_max_epochs = args.num_max_epochs
        self.learning_rate = args.learning_rate

        # TODO data/glove loading....


        # Load data
        loader = DataLoader(
            data_path = '../data/',
            train_filename = 'train.tsv.zip',
            pad_size = 10)
        loader.read_data()
        self.num_class = loader.num_class
        self.vocab = loader.vocab
        self.vocab_size = len(loader.vocab)

        # Data iterators
        self.train_iter = PaddingDatawithTarget(loader.train)
        self.test_iter = PaddingDatawithTarget(loader.test)

        # Load glove
        if self.load_glove:
            self.emb = load_glove(
                emb_path = '../data/glove.6B/',
                emb_filename= 'glove.6B.300d.txt', # 'test.txt', #
                vocab = self.vocab,
                emb_size = self.emb_size)
            self.emb_size = self.emb.shape[1]


        self.sess = None



    def build_graph(self):

        # create sesion
        self.sess = tf.Session()

        # Plaeholders(keep data)
        self.x = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None])  # [B x L]
        self.y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        self.seqlen = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        keep_prob = tf.constant(1.0)

        # Embedding layer (normal nums turn into vector with lenth of hidden_size
        # TODO double check this
        self.WW = tf.Variable(tf.constant(np.random.rand(self.vocab_size, self.emb_size).astype(np.float32)), trainable=self.emb_trainable, name='WW')
        self.embedding_placeholder = tf.placeholder(tf.float32, shape=[self.vocab_size, self.emb_size])
        self.embedding_load = self.WW.assign(self.embedding_placeholder)

        rnn_inputs = tf.nn.embedding_lookup(self.WW, self.x)  # [B x S x 1] x [V x H] -> [B x S x H]

        # RNN: [B x S x H] -> final_state [B x H]
        cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        init_state = tf.get_variable('init_state', [1, self.hidden_size])  # , initializer=tf.constant_initializer(0.0))
        init_state = tf.tile(init_state, [self.batch_size, 1])  # init * [batch_size,1], make same initial state for all batches
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=self.seqlen, initial_state=init_state)

        # Dropouts
        # TODO double check
        # rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob=keep_prob)

        # Softmax layer with last output: [B x H] [ H x N ] = [B x N]
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [self.hidden_size, self.num_class])
            b = tf.get_variable('b', [self.num_class])

        self.logits = tf.matmul(final_state, W) + b
        self.preds = tf.nn.softmax(self.logits)
        self.correct = tf.equal(tf.cast(tf.argmax(self.preds, 1), tf.int32), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        # precision & recall
        self.preds_num = tf.cast(tf.argmax(self.preds, 1), tf.int32)
        # loss([B x N], y[B x N])
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)




    def train(self, verbose=False):

        self.sess.run(tf.global_variables_initializer())

        if self.load_glove:
            self.sess.run([self.embedding_load], feed_dict={self.embedding_placeholder: self.emb})
            if verbose:
                print 'WW', self.sess.run(self.WW[0][0:5]), 'emb', self.emb[0][0:5]


        step, accuracy, loss = 0, 0, 0.
        train_losses, test_losses = [], []
        test_CM = np.zeros(shape =[self.num_class,self.num_class])
        test_precision, te_recall, te_fmeasure = [],[],[]
        current_epoch = 0
        num_batch_train = self.train_iter.size / self.batch_size

        # start epochs here
        while current_epoch < self.num_max_epochs:
            step += 1
            batch = self.train_iter.next_batch(self.batch_size)
            # print batch[0], batch[1], batch[2]
            accuracy_,loss_, _ = self.sess.run([self.accuracy, self.loss, self.train_step],
                                          feed_dict={self.x: batch[0], self.y: batch[1], self.seqlen: batch[2]})
            # , g['dropout']: 0.7}

            accuracy += accuracy_
            loss += loss_

            if step % 100 == 0:
                print "[%d/%d] Train Loss %.4f, Accuracy %.4f "%(step, num_batch_train, 1.0*loss/step, 100.0*accuracy/step)

            if self.train_iter.epochs > current_epoch:
                current_epoch += 1
                train_losses.append(accuracy / step)
                step, accuracy = 0, 0

                te_epoch = self.test_iter.epochs
                while self.test_iter.epochs == te_epoch:
                    step += 1
                    batch = self.test_iter.next_batch(self.batch_size)
                    # print batch[0], batch[1], batch[2]
                    accuracy_, preds_num_ = self.sess.run([self.accuracy, self.preds_num],
                                                     feed_dict={self.x:batch[0], self.y: batch[1], self.seqlen:batch[2]})

                    rnn_conf = confusion_matrix(preds_num_,batch[1],self.num_class)
                    accuracy += accuracy_
                    test_CM += np.array(rnn_conf)

                #print test_CM

                precision, recall, fmeasure = prec_rec(test_CM)
                test_losses.append(accuracy / step)
                test_precision.append(precision)
                te_recall.append(recall)
                te_fmeasure.append(fmeasure)
                step, accuracy, loss = 0, 0, 0.
                test_CM = np.zeros(shape=[self.num_class, self.num_class])
                print "\tAccuracy after epoch %d, train acc %.2f test acc %.2f" \
                      %(current_epoch, 100.0 * train_losses[-1], 100.0 * test_losses[-1]), test_precision[-1], te_recall[-1], te_fmeasure[-1]



    # def test(self):
    #     #TODO



"""
USAGE in shell script: run.sh

python week2_refactor.py \
    --batch_size 32 \

"""
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default = 128)
    parser.add_argument("--hidden_size", default = 256)
    parser.add_argument("--emb_size", default = 300)

    parser.add_argument("--emb_trainable", action='store_true',default=True)
    parser.add_argument("--load_glove", action='store_true',default=True)
    parser.add_argument("--num_max_epochs", default = 10)
    parser.add_argument("--learning_rate", default=5e-3)


    args = parser.parse_args()

    model = SentiRNN(args)
    model.build_graph()
    model.train()




