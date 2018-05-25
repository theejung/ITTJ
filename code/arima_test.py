import statsmodels.api as sm
import itertools
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = sm.datasets.co2.load_pandas()
y = data.data
y = y['co2'].resample('MS').mean()
y = y.fillna(y.bfill())

y_train = y[:480]
y_test = y[480:]
print y_test
print len(y), len(y_train), len(y_test)
#y.plot(figsize=(15, 6))
#plt.show()

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

#warnings.filterwarnings("ignore") # specify to ignore warning messages

aic = []
for param in pdq:
    try:
        print "START {}".format(param)
        mod = ARIMA(y_train,order=param)
        results = mod.fit()
        #print('ARIMA{} - AIC:{}'.format(param, results.aic))
        aic.append([param, results.aic])
        #print result.aic
    except:
        continue

print aic

mod = ARIMA(y_train, order=(1, 1, 0))
results = mod.fit()

#[[(0, 0, 0), 4483.106593047252],
# [(0, 0, 1), 3775.873381880716],
# [(0, 1, 0), 1676.8108550803245],
# [(0, 1, 1), 1374.168084697767],
# [(1, 0, 0), 1691.8735505960667],
# [(1, 0, 1), 1387.7120858137182],
# [(1, 1, 0), 1328.4721164054165],
# [(1, 1, 1), 1656.0240432815976]]
#
#[[(0, 0, 0), 3975.9201019009915],
# [(0, 0, 1), 3332.0588368723393],
# [(0, 1, 0), 1527.9470948428677],
# [(0, 1, 1), 1258.2698407281262],
# [(1, 0, 0), 1542.4370868855203],
# [(1, 0, 1), 1271.2952019088214],
# [(1, 1, 0), 1212.1260695322044],
# [(1, 1, 1), 1508.9602205645995]]

