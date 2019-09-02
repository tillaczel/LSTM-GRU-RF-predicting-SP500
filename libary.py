import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def import_data(file_name, column_of_Y):
    Data = pd.read_csv(file_name)
    #print(Data.isna().sum())
    Data.dropna(inplace=True)
    dates = Data.iloc[:, 0]
    Data = Data.drop(['Date'], axis=1)
    prices = Data.iloc[:, column_of_Y]
    return dates, Data, prices

def resample_data(df,frequency):
    df['Date'] = pd.to_datetime(df['Date'])

    df = (df.set_index('Date')
            .resample(frequency).first()
            .reset_index()
           .reindex(columns=df.columns)).dropna()
    return df

def log_return(Data):
    Returns = np.log(Data/Data.shift(1))
    Returns.dropna(inplace=True)
    return Returns

def inverse_log_return(prices, returns):
    returns = np.exp(returns)
    predicted_prices = np.multiply(np.reshape(prices, np.size(prices)), np.reshape(returns, np.size(returns)))
    return predicted_prices

def visualise_data(Data, Returns):
    Data.plot(subplots=True, legend=False)
    plt.show()
    Returns.plot(subplots=True, legend=False)
    plt.show()
    Returns.hist(bins=100)
    plt.show()
    Returns.boxplot()
    plt.show()
    plt.matshow(Returns.T, interpolation=None, aspect='auto', cmap='Greys')
    plt.show()