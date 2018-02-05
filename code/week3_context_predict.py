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
import pandas as pd
from numpy import newaxis
from sklearn.preprocessing import MinMaxScaler
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

from utils import rmse
from data_loader import TimeSeriesDataLoader,ContextDataLoader

class TimeSerisLSTM():
    def __init__(self):
        self.layers = args.layers #decide size of 2 LSTM layers
        self.dropout= args.dropout
        self.dropout_rt = args.dropout_rt
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.epoch = args.epoch

        self.step_size = args.step_size
        self.test_size = args.test_size
        self.window_size = args.window_size

        self.start_date = args.start_date
        self.end_date = args.end_date
        self.companies = args.companies
        self.timeseries_type = args.timeseries_type

        self.norm = args.norm

        # load context series
        cont_loader = ContextDataLoader(
            data_path = '../data/context/',
            ticker = self.companies,
            start_date = self.start_date,
            end_date = self.end_date,
            seq_len = self.window_size,
            step = self.step_size,
            normalize = self.norm)
        context = cont_loader.load_data()

        # load time series
        ts_loader = TimeSeriesDataLoader(
            data_path = '../data',
            filename = 'daily_price_5comp.csv', #'stock_vol_all.csv',
            feature = [self.timeseries_type],
            ticker = self.companies,
            start_date = self.start_date,
            end_date = self.end_date,
            seq_len = self.window_size,
            step = self.step_size,
            normalize = self.norm)

        series = ts_loader.load_data()

        if self.norm:
          self.scaler = MinMaxScaler(feature_range=(0, 1))
          self.scaler = self.scaler.fit(series)
          series_norm = self.scaler.transform(series)
          series_norm = pd.DataFrame([x[0] for x in series_norm], columns=series.columns)
          series_norm.index = series.index
        else:
          series_norm = series

        # concatenate two vectors
        df = context.join(series_norm, how='inner')
        #df = series
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
          t_x = test[idx: idx + self.window_size]
          t_y = test[idx + self.window_size : idx + self.window_size + self.step_size]
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
            return_sequences=False))
        if self.dropout:
            self.model.add(Dropout(self.dropout_rt))

        self.model.add(Dense(output_dim = self.feature))
        #self.model.add(Activation("linear"))

        start = time.time()
        self.model.compile(loss="mse", optimizer="rmsprop",
            lr=self.learning_rate)
        print("Compilation Time: ", time.time() - start)



    def predict(self, filename = 'result.txt'):

        y = []
        y_pred = []

        self.prediction = []
        for i in range(len(self.test_x)):
            curr_frame = self.test_x[i]
            pred = self.model.predict(curr_frame[newaxis,:,:])
            self.prediction.append(pred[0,-1])

            if not self.norm:
              y.append(self.test_y[i,-1])
              y_pred.append(pred[0,-1])
              print self.test_x[i,:,-1],y[-1], y_pred[-1]
            else:
              test_x_inverse = self.scaler.inverse_transform(self.test_x[i,:,-1])
              y.append(self.scaler.inverse_transform([self.test_y[i,-1]])[0])
              y_pred.append(self.scaler.inverse_transform([pred[0,-1]])[0])
              print test_x_inverse,y[-1], y_pred[-1]

        r = rmse(y,y_pred)
        print 'RMSE:', r
        with open(filename, 'a') as fout:
          fout.write('%s\t%s\t%.4f\n'%(
            self.companies, self.timeseries_type,r ))


    def train(self, verbose = False):
        self.model.fit(
            self.train_x,
            self.train_y,
            batch_size = self.batch_size,
            epochs=self.epoch,
            validation_split = 0.05)


#TODO (1) model save/restore, (2) train/test loading structure efficiently, (3) RMSE calculation (4) visualziation
#TODO (5) add stock price, (6) Apple -> 10 companies, more periods, (7) Task - 3. (random word vector).

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default=[200,200])
    parser.add_argument("--dropout",action='store_true',default=True)
    parser.add_argument("--dropout_rt", default = 0.2)
    parser.add_argument("--learning_rate", default = 0.5) #0.5 for price 0.001 for volatil


    parser.add_argument("--epoch", default=300)
    parser.add_argument("--batch_size", default= 5)

    parser.add_argument("--norm", action='store_true', default=False)

    #Hyper-parameter for data preparation
    parser.add_argument("--test_size", default=252)
    parser.add_argument("--step_size", default=1)
    parser.add_argument("--window_size", default=30)


    parser.add_argument("--start_date", default="2009-01-01")
    parser.add_argument("--end_date", default="2013-12-31")

    parser.add_argument("--companies", default="GOOG")
    # ['AAPL', 'AMZN', 'MSFT', 'FB', 'GOOG']
    parser.add_argument("--timeseries_type", default="last_price")
    # last_price volatil


    args = parser.parse_args()

    mymodel = TimeSerisLSTM()
    mymodel.build_graph()
    mymodel.train()
    mymodel.predict()

