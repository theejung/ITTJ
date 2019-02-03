import os,sys
import pandas as pd
import zipfile
import collections
from tensorflow.contrib import learn
import numpy as np

from utils import lowerr


class ContextDataLoader():
https://github.com/oscarg933/ITTJ/tree/https/github.com/oscarg933/jquery-ui-touch-punch/tree/%2523tabs
    def __init__(self, data_path, ticker="AAPL", start_date="1990-01-01", end_date="2010-12-31", seq_len = 12, step = 1, normalize = False):
        self.data_path = data_path
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.seq_len = seq_len
        self.step = step
        self.normalize = normalize
        #self.test_size = test_size

    def load_data(self):
        vecs = []
        dates = []
        for line in open(os.path.join(self.data_path, '%s.context'%(self.ticker)),'r'):
            tks = line.strip().split('\t')
            date = tks[0]
            dates.append(date)
            vec = [float(v) for v in tks[1].strip().split(' ')]
            vecs.append(vec)
        columns = ['v%d'%(i) for i in range(len(vecs[0]))]

        df = pd.DataFrame(np.array(vecs), columns= columns)
        df.index = pd.to_datetime(dates)
        df = df[self.start_date:self.end_date]
        #df = df.to_frame()

        # filter my first day of each month
        df.index = df.index.astype('str')
        df = df[df.index.str.contains('-[0-9]*')]
        print(len(df))

        return df

#         # 3. Make a 3-dim array [N x W x F] N = num of train, W = seq length, F = num of features
        # df_array = np.array(df).transpose()
        # sequence_length = self.seq_len + self.step
        # print self.seq_len, self.step


        # #

        # for f in df_array: #what if we have several features?
            # result = list()
            # #print len(f) - sequence_length
            # for i in xrange(len(f) - sequence_length+1):
                # #print f[i:i+sequence_length]
                # result.append(f[i:i+sequence_length])

            # result = np.array(result)
            # print len(result)

            # train_size = len(result) -self.test_size
            # train, test = result[:train_size], result[train_size:]
            # self.train_x, self.train_y = train[:,:-1*self.step], train[:,-1*self.step:]
            # self.test_x, self.test_y = test[:,:-1*self.step], test[:,-1*self.step:]

            # print len(self.train_x), len(self.train_y), len(self.test_x), len(self.test_y)

            # # 4. Normalize window if normalize_window = True
            # if self.normalize:
                # self.normalize_window(self.train_x)
                # self.normalize_window(self.test_x)

            # self.train_x = np.reshape(self.train_x, (self.train_x.shape[0], self.train_x.shape[1]))
            # self.test_x = np.reshape(self.test_x, (self.test_x.shape[0], self.test_x.shape[1]))
            # print self.train_x.shape, self.test_x.shape

        # import pdb; pdb.set_trace()


#    def normalize_window(self, arr):
#        array_change = []
#
#        for list in arr:
#            array_change.append([(float(p)/list[0]) - 1 for p in list])
#
#        arr = array_change
#




class TimeSeriesDataLoader():

    def __init__(self, data_path, filename, feature = ['volatil'], ticker="AAPL", start_date="1990-01-01", end_date="2010-12-31", seq_len = 12, step = 1, normalize = False, is_batch=True):
        self.data_path = data_path
        self.filename = filename
        self.feature = feature
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.seq_len = seq_len
        self.step = step
        self.normalize = normalize
        #self.test_size = test_size
        #self.is_batch = is_batch

    def load_data(self):
        # 1. Load all CSV file to pandas Dataframe, indexing with timestamp
        df = pd.read_csv(os.path.join(self.data_path, self.filename), header = 0)

        #print df.head(), df['ym'].head()
        df.index = pd.to_datetime(df['ym'])

        # 2. Left df only specific company and period
        df = df.loc[df['ticker'] == self.ticker]
        df = df[self.start_date:self.end_date]
        df = df[self.feature]

        return df

#        train_size = len(df) - self.test_size
#        train, test = df[:train_size], df[train_size:]
#
#        # 3. Make a 3-dim array [N x W x F] N = num of train, W = seq length, F = num of features
#        df_array = np.array(df[self.feature]).transpose()
#        sequence_length = self.seq_len + self.step
#        print self.seq_len, self.step
#
#        for f in df_array: #what if we have several features?
#            result = list()
#            #print len(f) - sequence_length
#            for i in xrange(len(f) - sequence_length+1):
#                #print f[i:i+sequence_length]
#                result.append(f[i:i+sequence_length])
#
#            result = np.array(result)
#            print len(result)
#
#            train_size = len(result) -self.test_size
#            train, test = result[:train_size], result[train_size:]
#            self.train_x, self.train_y = train[:,:-1*self.step], train[:,-1*self.step:]
#            self.test_x, self.test_y = test[:,:-1*self.step], test[:,-1*self.step:]
#
#            print len(self.train_x), len(self.train_y), len(self.test_x), len(self.test_y)
#            print self.train_x[0], self.train_y[0], self.train_x[1], self.train_y[1]
#
#            # 4. Normalize window if normalize_window = True
#
#            if self.normalize:
#                self.normalize_window(self.train_x)
#                self.normalize_window(self.test_x)
#
#            self.train_x = np.reshape(self.train_x, (self.train_x.shape[0], self.train_x.shape[1],len(self.feature)))
#            self.test_x = np.reshape(self.test_x, (self.test_x.shape[0], self.test_x.shape[1], len(self.feature)))
#
#
#
#
#    def normalize_window(self, arr):
#        array_change = []
#
#        for list in arr:
#            array_change.append([(float(p)/list[0]) - 1 for p in list])
#
#        arr = array_change
#




class MRDataLoader():
    def __init__ (self, data_path, train_filename='train.tsv.zip', pad_size=10, max_vocab=False):
        self.data_path = data_path
        self.train_filename = train_filename
        self.pad_size = pad_size  # based on average length of sentences: 6.89 #TODO get from data late
        self.max_vocab = max_vocab
        self.train = []
        self.test = []
        self.num_class = -1




    def read_data_text(self, only_test=False):

        # road train data into pandas DataFrame
        with zipfile.ZipFile(os.path.abspath(self.data_path + self.train_filename)) as z:
            with z.open(z.namelist()[0]) as f:
                train = pd.read_csv(f, header=0, delimiter='\t')
        print 'Load train data done!'
        train_sentence, train_length = lowerr(train.Phrase)
        train_y = train.Sentiment

        ## Train vs Test
        df = pd.DataFrame({'X': train_sentence, 'Y': train_y, 'length': train_length})
        df = df.sample(frac=1, random_state=63).reset_index(drop=True)
        test_len = np.floor(len(df) * 0.1)
        self.test, self.train = df.ix[:test_len - 1], df.ix[test_len:]
        print self.train['X'].shape, self.train['Y'].shape
        print self.test['X'].shape, self.test['Y'].shape




    def read_data(self):
        # road train data into pandas DataFrame
        with zipfile.ZipFile(os.path.abspath(self.data_path + self.train_filename)) as z:
            with z.open(z.namelist()[0]) as f:
                train = pd.read_csv(f, header=0, delimiter='\t')
        print 'Load train data done!'

        train_sentence, self.length = lowerr(train.Phrase)
        self.create_vocab(train_sentence)
        self.y = train.Sentiment

        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=self.pad_size)
        vocab_processor.fit(self.vocab)
        X = list(vocab_processor.fit_transform(train_sentence))
        print 'X transformation Done!'

        ## Train vs Test
        df = pd.DataFrame({'X': X, 'Y': self.y, 'length': self.length})
        df = df.sample(frac=1, random_state=63).reset_index(drop=True)
        test_len = np.floor(len(df) * 0.1)
        self.test, self.train = df.ix[:test_len - 1], df.ix[test_len:]
        self.num_class = len(np.unique(df.Y))
        print 'Train/Test split Done!', len(X)


    def create_vocab(self,train_sentence):
        # Top n vocab only!
        # n = 20000  # We can change it if we want!
        flat_sentences = [z for y in [x.strip().split(' ') for x in train_sentence] for z in y]
        #print 'Number of Words in Train', len(flat_sentences)
        #print flat_sentences[:100]

        freq_dict = collections.Counter(flat_sentences)
        print 'Create Vocab: Top 10 freq words and freq', freq_dict.most_common(10)
        #print '1st, 2nd, 3rd freq Words', freq_dict.most_common(100)[0][0], freq_dict.most_common(100)[1][0], freq_dict.most_common(100)[2][0]

        if self.max_vocab:
            freq_list = freq_dict.most_common(self.max_vocab)
        else:
            freq_list = freq_dict.most_common()
        self.vocab = [k for k,c in freq_list]

        print "Create Vocab Done!", len(self.vocab)


class S140DataLoader():
    def __init__ (self, data_path,
        train_filename='training.1600000.processed.noemoticon.csv',
        test_filename='testdata.manual.2009.06.14.csv',
        pad_size=10, max_vocab=False):

        self.data_path = data_path
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.pad_size = pad_size  # based on average length of sentences: 6.89 #TODO get from data late
        self.max_vocab = max_vocab
        self.train = []
        self.test = []
        self.num_class = -1


    def read_data_text(self, only_test=False):

        if not only_test:
          # road train data into pandas DataFrame
          with open(os.path.abspath(self.data_path + self.train_filename)) as f:
              train = pd.read_csv(f, header=None, delimiter=',', usecols=[0,5])
          print 'Load train data done!'
          train_sentence, train_length = lowerr(train[5])
          train_y = train[0]
          self.train = pd.DataFrame({'X': train_sentence, 'Y': train_y, 'length': train_length})
          print self.train['X'].shape, self.train['Y'].shape

        # road train data into pandas DataFrame
        with open(os.path.abspath(self.data_path + self.test_filename)) as f:
            test = pd.read_csv(f, header=None, delimiter=',', usecols=[0,5])
        print 'Load test data done!'
        test_sentence, test_length = lowerr(test[5])
        test_y = test[0]
        self.test = pd.DataFrame({'X': test_sentence, 'Y': test_y, 'length': test_length})
        print self.test['X'].shape, self.test['Y'].shape





    def read_data(self):
        # road train data into pandas DataFrame
        with open(os.path.abspath(self.data_path + self.train_filename)) as f:
            train = pd.read_csv(f, header=None, delimiter=',', usecols=[0,5])
        print 'Load train data done!'

        # road train data into pandas DataFrame
        with open(os.path.abspath(self.data_path + self.test_filename)) as f:
            test = pd.read_csv(f, header=None, delimiter=',', usecols=[0,5])
        print 'Load test data done!'

        def processor(sents, vocab, pad_size):
          sents_vec = []
          for sent in sents:
            sent_vec = np.zeros(pad_size, dtype=np.int32)
            for widx,w in enumerate(sent.split()[:pad_size]):
              if w in vocab:
                sent_vec[widx] = vocab[w]
            sents_vec.append(sent_vec)
          return sents_vec

        # Train
        train_sentence, train_length = lowerr(train[5])
        self.create_vocab(train_sentence)
        train_y = train[0]
        train_x = processor(train_sentence, self.vocab_dic, self.pad_size)
        self.num_class = len(np.unique(train_y))
        print 'Unique y in train', len(np.unique(train_y)),np.unique(train_y)

        Y_vocab = list(np.unique(train_y))
        y_vecs = []
        for yone in train_y:
          y_vecs.append(Y_vocab.index(yone))
        train_y = y_vecs
        self.train = pd.DataFrame({'X': train_x, 'Y': train_y, 'length': train_length})


        def filter_neutral(sents,lens,ys):
          new_sents, new_lens, new_ys = [], [] ,[]
          for sent,leng,y in zip(sents,lens, ys):
            if y == '2' or y == 2: continue
            else:
              new_sents.append(sent)
              new_lens.append(leng)
              new_ys.append(y)
          return new_sents, new_lens, new_ys


        ## Test
        t_sentence, t_length = lowerr(test[5])
        t_y = test[0]
        test_sentence, test_length, test_y = filter_neutral(t_sentence, t_length, t_y)
        print 'Unique y in test', len(np.unique(test_y)),np.unique(test_y)
        y_vecs = []
        for yone in test_y:
          y_vecs.append(Y_vocab.index(yone))
        test_y = y_vecs
        test_x = processor(test_sentence, self.vocab_dic, self.pad_size)
        self.test = pd.DataFrame({'X': test_x, 'Y': test_y, 'length': test_length})


        print 'Train/Test split Done!'
        print self.train['X'].shape, self.train['Y'].shape
        print self.test['X'].shape, self.test['Y'].shape


    def create_vocab(self,train_sentence):
        # Top n vocab only!
        # n = 20000  # We can change it if we want!
        flat_sentences = [z for y in [x.strip().split(' ') for x in train_sentence] for z in y]
        #print 'Number of Words in Train', len(flat_sentences)
        #print flat_sentences[:100]

        freq_dict = collections.Counter(flat_sentences)
        print 'Create Vocab: Top 10 freq words and freq', freq_dict.most_common(10)
        #print '1st, 2nd, 3rd freq Words', freq_dict.most_common(100)[0][0], freq_dict.most_common(100)[1][0], freq_dict.most_common(100)[2][0]

        if self.max_vocab:
            freq_list = freq_dict.most_common(self.max_vocab)
        else:
            freq_list = freq_dict.most_common()
        self.vocab = ['UNK']
        self.vocab += [k for k,c in freq_list]
        self.vocab_dic = {w:i for i,w in  enumerate(self.vocab)}

        print "Create Vocab Done!", len(self.vocab)




