# Install pandas-datareader, an up-to-date remote data access for pandas
#!pip install git+https://github.com/pydata/pandas-datareader.git

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from datetime import datetime, timedelta
from sklearn.cluster import KMeans

def get_data(ticker, time_window=1000):
    """
    Uses pandas-datareader to load daily stock data and returns a DataFrame.
    
    ticker: str, stock symbol which identifies shares of a stock
    time_window: int, number of days to be considered, to date (default = 1000)
    """
    ticker = ticker.upper()
    end_date = datetime.today()
    start_date = end_date - timedelta(days = time_window)
    
    # convert objects to a string according to a given format
    end_date = end_date.strftime('%Y-%m-%d')
    start_date = start_date.strftime('%Y-%m-%d')
    
    # feed web-data into a pandas DataFrame
    df = web.DataReader(ticker, 'yahoo', start_date, end_date)
    
    return df

def plot_trend(x, y, labels, ticker):
    """
    Scatter plot the data.
    
    x: ndarray, evenly spaced values within the interval (0, len(y))
    y: ndarray, closing prices
    labels: ndarray, labels of each point after K-Means clustering
    ticker: str, stock symbol which identifies shares of a stock
    """
    colors = ['tab:red','tab:orange','tab:green']
    nclass = len(np.unique(labels))
    
    plt.figure(figsize=(15,5))
    for i in range(nclass):
        xx = x[labels == i]
        yy = y[labels == i]
        plt.scatter(xx, yy, c=colors[i], s=3, label=i)
        
    plt.legend(frameon=False, fontsize='medium')
    plt.title(f'{ticker.upper()}')
    plt.savefig(f'{ticker.upper()}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def get_trend(c, window=20):
    '''
    Filters the time series with a forward-backward EMA and fits a K-Means clustering algorithm.
    Returns data points and labels for use with the plot_trend() function.
    
    c: ndarray, daily close prices
    window: int, lookback window for smoothing
    '''
    
    # forward-backward filtering with EMA
    cs = pd.Series(c)
    f_ema = cs.ewm(span=window).mean()
    fb_ema = f_ema[::-1].ewm(span=window).mean()[::-1]
    
    # evaluate log-return and fit the clustering model
    lr = np.diff(np.log(fb_ema.values))        
    km = KMeans(3).fit(lr.reshape(-1,1))
    lb = km.labels_
    
    # change the labels to have some semblance of order
    cc = km.cluster_centers_.flatten()
    temp = [(cc[i], i) for i in range(3)]
    temp = sorted(temp, key = lambda x: x[0])
    
    labels = np.zeros(len(lb), dtype = int)
    for i in range(1,3):
        old_lb = temp[i][1]
        idx = np.where(lb == old_lb)[0]
        labels[idx] = i
    
    x = np.arange(len(labels))
    y = fb_ema.values[1:]
    
    return x, y, labels