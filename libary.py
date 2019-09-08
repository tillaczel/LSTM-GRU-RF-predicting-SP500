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

def creating_study_periods(frequencies, frequencies_number_of_samples, frequency_index):
    # Import data
    Data = pd.read_csv('Data_1min.csv', dtype=str)
    Data = resample_data(Data,frequencies[frequency_index])

    # Create datasets
    dates = Data['Date']
    Data.drop('Date', inplace=True, axis=1)

    Data = Data.astype('float64')

    # Log return
    Returns = log_return(Data).to_numpy()

    # Create study peridos
    number_of_study_periods = np.floor(Returns.shape[0]/frequencies_number_of_samples[frequency_index]).astype(int)-1
    study_periods = np.zeros((2,number_of_study_periods, frequencies_number_of_samples[frequency_index]*2))
    for i in range(number_of_study_periods):
        study_periods[0,i] = Returns[i*frequencies_number_of_samples[frequency_index]:\
                                        (i+2)*frequencies_number_of_samples[frequency_index]].flatten()
        study_periods[1,i] = dates.iloc[i*frequencies_number_of_samples[frequency_index]:\
                                        (i+2)*frequencies_number_of_samples[frequency_index]].to_numpy().flatten()
    
    return number_of_study_periods, study_periods, Data, dates
    
def visualize_results(mse):
    fig = plt.figure(figsize=(14,8))
    plt.plot(mse)
    plt.title('MSE of models')
    plt.legend(['ARMA', 'LSTM', 'GRU', 'Zero return'])
    plt.show()

def visualize_data(Data, Returns):
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