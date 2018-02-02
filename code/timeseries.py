#!/usr/bin/python
#-*- coding: utf-8 -*-


import os, sys
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import datetime as dt
from datetime import timedelta

#from data_processor import load_data

reload(sys)
sys.setdefaultencoding('utf-8')


class TSModel:
    '''
    Classical time series models
        - AR
        - MA (TODO)
        - ARMA (TODO)
        - VAR
    '''

    def __init__(self, endog=None, feature=None, method='ar', steps=1, isnpa=False, verbose=False):

        self.endog = endog # 1-d pandas.DataFrame or numpy.array
        self.feature = feature # n-d pandas.DataFrame or numpy.array. Needed for VAR
        self.method = method # 'ar', 'ma', 'arma', 'var'
        self.steps = steps # number of steps to forecast
        self.isnpa = isnpa # flag for numpy.array
        self.verbose = verbose # for debug
        self.timedelta = None
        if not isnpa and len(endog.index) > 1:
            self.timedelta = endog.index[-1] - endog.index[-2]

        # result values
        self.result = {}
        self.result['fit_results'] = None # fit results
        self.result['params'] = None # fitted model parameters (= self.result['fit_results'].params)
        self.result['fit'] = None # fitted values
        self.result['forecast'] = None # 'self.steps'-ahead forecasted values


    def fit_forecast(self):
        if self.method == 'ar':
            self.__fit_forecast_AR()
        elif self.method == 'ma':
            self.__fit_forecast_MA()
        elif self.method == 'arma':
            self.__fit_forecast_ARMA()
        elif self.method == 'var':
            self.__fit_forecast_VAR()
        else:
            raise Exception("Invalid method: {0}".format(self.method))
        if self.result['fit_results'] is None:
            raise Exception("Fail to get a fit")
        if self.result['params'] is None:
            raise Exception("Fail to get model parameters")
        if self.result['fit'] is None:
            raise Exception("Fail to get fitted values")
        if self.result['forecast'] is None:
            raise Exception("Fail to get forecasted values")


    def __fit_forecast_AR(self):
        # Suppose that all data clearing is done outside
        if self.verbose:
            print "=== fit and forecast AR ==="

        ar_model = sm.tsa.AR(endog=self.endog, missing='drop')

        ## fit ##
        # TODO: find the best parameters - AR.fit(maxlag=None, method='cmle', ic=None, trend='c' ...)
        maxlags = min(int(12*(len(self.endog)/100.)**(1./4)), 10)
        ar_fit = ar_model.fit(maxlag=maxlags, method='cmle', ic='bic', trend='nc')

        ## forecast ##
        # TODO: missing date handling if self.timedelta is not None
        #start = self.endog.index[-1] - ar_fit.k_ar * self.timedelta
        #end = self.endog.index[-1] + self.steps * self.timedelta
        start = len(self.endog) - ar_fit.k_ar
        end = len(self.endog) - 1 + self.steps

        #print len(self.endog), ar_fit.k_ar, start, end, self.steps
        #print self.endog.shape
        ar_forecast = ar_fit.predict(start, end)
        #ar_forecast = ar_fit.predict(len(self.endog),len(self.endog) + self.steps)

        ## assign results ##
        self.result['fit_results'] = ar_fit
        self.result['params'] = ar_fit.params
        if self.isnpa:
            self.result['fit'] = np.append(self.endog[:ar_fit.k_ar], ar_fit.fittedvalues)
            self.result['forecast'] = ar_forecast[-self.steps:]
        else:
            self.result['fit'] = np.append(self.endog.values[:ar_fit.k_ar], ar_fit.fittedvalues.values)
            self.result['forecast'] = ar_forecast.values[-self.steps:]

        if self.verbose:
            print "lag length:{0}, # of trend terms:{1}".format(ar_fit.k_ar, ar_fit.k_trend)
            print "model parameters:\n", ar_fit.params
            print "t-values associated with 'params':\n", ar_fit.tvalues
            print "fitted values:\n", ar_fit.fittedvalues
            print "residuals:\n", ar_fit.resid
            print "variance of residuals:", ar_fit.sigma2


    def __fit_forecast_VAR(self):
        if self.verbose:
            print "=== fit and forecast VAR === "
        
        if self.isnpa:
            data = np.array([self.endog, self.feature], dtype=float).transpose()
        else:
            data = pd.concat([self.endog, self.feature], axis=1, join='inner')
        var_model = statsmodels.tsa.api.VAR(endog=data, missing='drop')

        ## fit ##
        # TODO: find the best input parameters - VAR.fit(maxlags=None, method='ols', ic=None, trend='c', ...)
        maxlags = min(int(12*(len(self.endog)/100.)**(1./4)), 10)
        #print maxlags
        #print data
        var_fit = var_model.fit(maxlags=maxlags, method='ols', ic=None, trend='nc')

        ## forecast ##
        if self.isnpa:
            npdata = data
        else:
            npdata = data.values
        var_forecast = var_fit.forecast(npdata[-var_fit.k_ar:], self.steps)

        ## assign results ##
        self.result['fit_results'] = var_fit
        self.result['params'] = var_fit.params
        # Here we are interested in only endog's values...
        if self.isnpa:
            self.result['fit'] = np.append(self.endog[:var_fit.k_ar], var_fit.fittedvalues[:,0])
        else:
            self.result['fit'] = np.append(self.endog.values[:var_fit.k_ar], var_fit.fittedvalues.values[:,0])
        self.result['forecast'] = var_forecast[:,0]



        self.result['model'] = var_fit


        if self.verbose:
            import traceback
            try:
                print self.result['fit_results'].summary()
            except Exception  as e:
                print (traceback.format_exc())


    def __fit_forecast_MA(self):
        if self.verbose:
            print "=== fit and forecast MA is not implemented yet === "
        self.result['fit_results'] = None
        self.result['params'] = None
        self.result['fit'] = None
        self.result['forecast'] = None


    def __fit_forecst_ARMA(self):
        if self.verbose:
            print "=== fit and forecast ARMA is not implemented yet=== "
        self.result['fit_results'] = None
        self.result['params'] = None
        self.result['fit'] = None
        self.result['forecast'] = None


if __name__ == "__main__":

    myTSModel = TSModel(isnpa=True, verbose=False)

    print ('## test AR with trivious data ##')
    myTSModel.endog = np.array([0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    myTSModel.fit_forecast()
    print ('1-step ahead forecasted value: {0}'.format(myTSModel.result['forecast']))

    print ('## test VAR with trivious data ##')
    myTSModel.method='var'
    myTSModel.endog = np.array([0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    myTSModel.feature = np.array([1.125, 1.25, 1.5, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025, 2049, 4097, 8193])
    myTSModel.fit_forecast()
    print ('1-step ahead forecasted value: {0}'.format(myTSModel.result['forecast']))

#    print ('## test AR with real data ##')
#    myTSModel.method='ar'
#    df = load_data('MemePhr', os.path.join(os.path.dirname(__file__),os.path.pardir,'data','ksc','MemePhr.txt'))
#    myTSModel.endog = np.array(df['Val'][0], dtype=float)
#    myTSModel.fit_forecast()
#    print ('1-step ahead forecasted value: {0}'.format(myTSModel.result['forecast']))

