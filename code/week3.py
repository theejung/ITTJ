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


from data_loader import TimeSeriesDataLoader

class TimeSerisLSTM():
    def __init__(self):
        self.layers = args.layers #decide size of 2 LSTM layers
        self.dropout= args.dropout
        self.dropout_rt = args.dropout_rt
        self.batch_size = args.batch_size
        self.epoch = args.epoch

        self.step_size = args.step_size
        self.window_size = args.window_size

        #TODO self.feature = args.feature


        loader = TimeSeriesDataLoader(
            data_path = '../data',
            filename = 'stock_vol.csv',
            feature = ['volatil'],
            ticker = 'AAPL',
            start_date = '1990-01',
            end_date = '2016-12',
            seq_len = self.window_size,
            step = self.step_size,
            normalize = False,
            test_size = 12)

        loader.load_data()

        self.feature = len(loader.feature)
        self.seq_len = loader.seq_len
        self.train_x = loader.train_x
        self.train_y = loader.train_y
        self.test_x = loader.test_x
        self.test_y = loader.test_y
        import pdb; pdb.set_trace()

   #loader.train_x, loader.train_y.shape, loader.test_x.shape, loader.test_y.shape

    def build_graph(self):
        self.model = Sequential()
        self.model.add(LSTM(
            output_dim =self.layers[0],
            input_dim =self.feature,
            return_sequences=True)) #Starting Layer! first layer size is coming from num of features that we have

        if self.dropout:
            self.model.add(Dropout(self.dropout_rt))

        self.model.add(LSTM(
            self.layers[1],
            return_sequences=False)) #Add one more LSTM Layer ->Only need hidden/cell size!

        if self.dropout:
            self.model.add(Dropout(self.dropout_rt))

        self.model.add(Dense(
            output_dim = self.layers[2]))

        self.model.add(Activation("linear"))

        start = time.time()
        self.model.compile(loss="mse", optimizer="rmsprop")
        print("Compilation Time: ", time.time() - start)

    def predict(self):
        self.prediction_seqs = []
        for i in range(int(len(self.test_x))/self.step_size):
            curr_frame = self.test_x[i * self.step_size]

            predicted = []

            for j in range(self.step_size):
                print i,j, curr_frame[newaxis,:,:]
                predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0]) #
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame,[self.seq_len-1], predicted[-1], axis=0) #Adding up 'predicted' value!
                print predicted, curr_frame

            self.prediction_seqs.append(predicted)

    def train(self, verbose = False):
        self.model.fit(
            self.train_x,
            self.train_y,
            batch_size = self.batch_size,
            epochs=self.epoch,
            validation_split = 0.05)

        self.predict()
        print self.test_y, self.prediction_seqs


#TODO (1) model save/restore, (2) train/test loading structure efficiently, (3) RMSE calculation (4) visualziation
#TODO (5) add stock price, (6) Apple -> 10 companies, more periods, (7) Task - 3. (random word vector).
#TODO priority: 3 2 6 7 1 4 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default=[100,200,1])
    parser.add_argument("--dropout",action='store_true',default=True)
    parser.add_argument("--dropout_rt", default = 0.2)
    parser.add_argument("--batch_size", default= 32)
    parser.add_argument("--epoch", default = 1)

    #Hyper-parameter for data preparation
    parser.add_argument("--step_size", default=1)
    parser.add_argument("--window_size", default=12)
    args = parser.parse_args()

    mymodel = TimeSerisLSTM()
    mymodel.build_graph()
    mymodel.train()

#[[0.00137618],[0.00117679],[0.00115418],[0.00185941],[0.00142668],[0.00074926],[0.00131295],[0.00070778],[0.0022864][0.00222184],[0.00116946]]
#[[0.00098015077], [0.00096921431], [0.00097124733], [0.00096955721], [0.000975893], [0.00098021922], [0.00097770256], [0.00097344629], [0.00096385967], [0.00096935942], [0.00097717589]]
