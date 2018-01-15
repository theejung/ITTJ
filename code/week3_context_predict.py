"""""""""""
Study LSTM!
http://colah.github.io/posts/2015-08-Understanding-LSTMs/
http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction
https://en.wikipedia.org/wiki/Long_short-term_memory
https://www.quora.com/In-LSTM-how-do-you-figure-out-what-size-the-weights-are-supposed-to-be
https://www.quora.com/How-is-the-hidden-state-h-different-from-the-memory-c-in-an-LSTM-cell
"""""""""""
import os, sys, argparse
import time
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


from data_loader import TimeSeriesDataLoader,ContextDataLoader

class TimeSerisLSTM():
    def __init__(self):
        self.layers = args.layers #decide size of 2 LSTM layers
        self.dropout= args.dropout
        self.dropout_rt = args.dropout_rt
        self.batch_size = args.batch_size
        self.epoch = args.epoch

        self.step_size = args.step_size

        self.test_size = args.test_size
        self.window_size = args.window_size

        self.start_date = '2009-01-01'
        self.end_date = '2013-12-31'
        self.companies = 'AAPL' # ['AAPL', 'AMZN', 'MSFT', 'FB', 'GOOG']

        # load context series
        cont_loader = ContextDataLoader(
            data_path = '../data/context/',
            ticker = self.companies,
            start_date = self.start_date,
            end_date = self.end_date,
            seq_len = self.window_size,
            step = self.step_size,
            normalize = False,
            test_size = self.test_size)
        context = cont_loader.load_data()

        # load time series
        ts_loader = TimeSeriesDataLoader(
            data_path = '../data',
            filename = 'stock_vol_all.csv',
            feature = ['last_price'],
            ticker = self.companies,
            start_date = self.start_date,
            end_date = self.end_date,
            seq_len = self.window_size,
            step = self.step_size,
            normalize = False,
            test_size = self.test_size,
            is_batch = False)
        series = ts_loader.load_data()

        # concatenate two vectors
        df = context.join(series, how='inner')
        print df.shape

        # create train/test
        self.create_batch(df)


    #Make a 3-dim array [N x W x F] N = num of train, W = seq length, F = num of features
    def create_batch(self, df):
        print df.shape
        self.feature = df.shape[1]

        # train: [N x F], test [ N x F]
        train_size = len(df) - self.test_size
        train, test = df[:train_size], df[train_size:]


        # train: [N x F], test [ N x F]
        train_x, train_y = [], []
        for idx in range(len(train) - self.window_size - self.step_size):
          t_x = train[idx: idx + self.window_size]
          t_y = train[idx + self.window_size : idx + self.window_size + self.step_size]
          train_x.append(t_x.as_matrix())
          train_y.append(t_y.as_matrix())


        test_x, test_y = [], []
        for idx in range(len(test) - self.window_size - self.step_size):
          t_x = train[idx: idx + self.window_size]
          t_y = train[idx + self.window_size : idx + self.window_size + self.step_size]
          test_x.append(t_x.as_matrix())
          test_y.append(t_y.as_matrix())

        self.train_x, self.train_y, self.test_x, self.test_y = \
            np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)
        self.train_y = np.reshape(self.train_y, (self.train_y.shape[0],-1))
        self.test_y = np.reshape(self.test_y, (self.test_y.shape[0],-1))
        print self.train_x.shape, self.train_y.shape, self.test_x.shape, self.test_y.shape



    def build_graph(self):
        self.model = Sequential()
        self.model.add(LSTM(
            output_dim =self.layers[0],
            input_dim = self.feature,
            return_sequences=True)) #Starting Layer! first layer size is coming from num of features that we have

        if self.dropout:
            self.model.add(Dropout(self.dropout_rt))

        self.model.add(LSTM(
            self.layers[1],
            return_sequences=False)) #Add one more LSTM Layer ->Only need hidden/cell size!

        if self.dropout:
            self.model.add(Dropout(self.dropout_rt))

        self.model.add(Dense(output_dim = self.feature))

        #self.model.add(Activation("linear"))

        start = time.time()
        self.model.compile(loss="mse", optimizer="rmsprop")
        print("Compilation Time: ", time.time() - start)



    def predict(self):
        self.prediction = []
        for i in range(len(self.test_x)):
            curr_frame = self.test_x[i]
            pred = self.model.predict(curr_frame[newaxis,:,:])
            self.prediction.append(pred[0,-1])
            print self.test_x[i,:,-1],self.test_y[i,-1], pred[0,-1]

    def train(self, verbose = False):
        self.model.fit(
            self.train_x,
            self.train_y,
            batch_size = self.batch_size,
            epochs=self.epoch,
            validation_split = 0.05)

        self.predict()


#TODO (1) model save/restore, (2) train/test loading structure efficiently, (3) RMSE calculation (4) visualziation
#TODO (5) add stock price, (6) Apple -> 10 companies, more periods, (7) Task - 3. (random word vector).
#TODO priority: 3 2 6 7 1 4 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default=[200,200])
    parser.add_argument("--dropout",action='store_true',default=True)
    parser.add_argument("--dropout_rt", default = 0.2)
    parser.add_argument("--batch_size", default= 5)
    parser.add_argument("--epoch", default = 100)

    #Hyper-parameter for data preparation

    parser.add_argument("--test_size", default=12)
    parser.add_argument("--step_size", default=1)
    parser.add_argument("--window_size", default=6)
    args = parser.parse_args()

    mymodel = TimeSerisLSTM()
    mymodel.build_graph()
    mymodel.train()

#[[0.00137618],[0.00117679],[0.00115418],[0.00185941],[0.00142668],[0.00074926],[0.00131295],[0.00070778],[0.0022864][0.00222184],[0.00116946]]
#[[0.00098015077], [0.00096921431], [0.00097124733], [0.00096955721], [0.000975893], [0.00098021922], [0.00097770256], [0.00097344629], [0.00096385967], [0.00096935942], [0.00097717589]]
