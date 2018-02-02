import os, sys, argparse
import numpy as np
import pandas as pd
from numpy import newaxis
from sklearn.preprocessing import MinMaxScaler
from timeseries import TSModel
from data_loader import TimeSeriesDataLoader, ContextDataLoader
from utils import rmse

class TimeSeriesAR():

    def __init__(self):
        self.step_size = args.step_size
        self.window_size = args.window_size
        self.model_type = args.model_type
        self.feature_type = args.feature_type

        self.start_date = args.start_date
        self.end_date = args.end_date
        self.companies = args.companies
        self.timeseries_type = args.timeseries_type
        self.norm = args.norm #TODO

        # load time series
        ts_loader = TimeSeriesDataLoader(
            data_path = '../data',
            filename = 'daily_price_5comp.csv', #'stock_vol_all.csv'
            feature = [self.timeseries_type],
            ticker = self.companies,
            start_date = self.start_date,
            end_date = self.end_date,
            seq_len = self.window_size,
            step = self.step_size,
            normalize = self.norm)

        series = ts_loader.load_data()

        if self.feature_type == 'context_simple':
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


            df = series.join(context, how = 'inner')
            series = df
            #import pdb;pdb.set_trace();

        if self.norm:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaler = self.scaler.fit(series)
            self.series_norm = self.scaler.transform(series)
            self.series_norm = pd.DataFrame([x[0] for x in series_norm], columns=series.columns)
            self.series_norm.index = series.index
        else:
            self.series_norm = series

        #Data preparation


    def forecasting(self):
        print 'Data len %.0f' %(len(self.series_norm))
        print 'Forecasting...%s %s' %(self.model_type, self.feature_type)
        y = self.series_norm.ix[:, 0]
        x = self.series_norm.ix[:, 1:]
        errsfit = []
        errsfor = []
        start = 0
        end = self.window_size
        while (end <= len(self.series_norm) - self.step_size):
            ytrain = y[start:end]
            ytest = y[end:end + self.step_size]
            fmodel = []

            if self.model_type == 'ar':
                #print ytrain.shape, ytest.shape
                # ytrain = sm.add_constant(ytrain)
                fmodel = TSModel(endog=ytrain, method=self.model_type, steps=self.step_size, isnpa=False, verbose=False)

            elif  self.model_type == 'var':
                xtrain = x[start:end]
                # feature filtering on segmented a constant variation.
                xtrain = xtrain.loc[:, (xtrain != xtrain.ix[0]).any()]
                fmodel = TSModel(endog=ytrain, feature=xtrain, method=self.model_type, steps=self.step_size, isnpa=False,
                                 verbose=False)

            # try:
            fmodel.fit_forecast()
            if fmodel.result is not None:
                efit = rmse(ytrain, fmodel.result['fit'])
                efor = rmse(ytest, fmodel.result['forecast'])
                # print '\t',model,start, end,  efit, efor , ytest,  fmodel.result['forecast'] #fmodel.result['fit'],
                errsfit.append(efit)
                errsfor.append(efor)
            start += self.step_size
            end += self.step_size

        print self.model_type, self.feature_type, self.companies, len(errsfit), len(errsfor), np.mean(errsfit), np.mean(errsfor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #Hyper-parameter for data preparation
    parser.add_argument("--step_size", default=1)
    parser.add_argument("--window_size", default=30)
    parser.add_argument("--model_type", default = 'var')
    parser.add_argument("--feature_type", default = 'context_simple')

    parser.add_argument("--norm", action='store_true', default=False)


    parser.add_argument("--start_date", default="2012-11-16") #"2012-11-16"
    parser.add_argument("--end_date", default="2013-12-31")

    parser.add_argument("--companies", default="GOOG")
    # ['AAPL', 'AMZN', 'MSFT', 'FB', 'GOOG']
    parser.add_argument("--timeseries_type", default="last_price")
    # last_price volatil


    args = parser.parse_args()

    mymodel = TimeSeriesAR()
    mymodel.forecasting()
