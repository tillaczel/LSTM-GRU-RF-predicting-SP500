import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

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
    Data = resample_data(Data, frequencies[frequency_index])

    # Create datasets
    dates = Data['Date']
    Data.drop('Date', inplace=True, axis=1)

    Data = Data.astype('float64')

    # Log return
    Returns = log_return(Data).to_numpy()

    # Create study peridos
    number_of_study_periods = np.floor(Returns.shape[0]/(frequencies_number_of_samples[frequency_index]/2)).astype(int)-3
    study_periods = np.zeros((2,number_of_study_periods, frequencies_number_of_samples[frequency_index]*2))
    for i in range(number_of_study_periods):
        study_periods[0,i] = Returns[(i*frequencies_number_of_samples[frequency_index]/2).astype(int):\
                                        ((i+4)*frequencies_number_of_samples[frequency_index]/2).astype(int)].flatten()
        study_periods[1,i] = dates.iloc[(i*frequencies_number_of_samples[frequency_index]/2).astype(int):\
                                        ((i+4)*frequencies_number_of_samples[frequency_index]/2).astype(int)].to_numpy().flatten()
    
    dates = dates[:((number_of_study_periods+3)*frequencies_number_of_samples[frequency_index]/2).astype(int)]
    return number_of_study_periods, study_periods, Data, dates
    
def visualize_results(mse):
    fig = plt.figure(figsize=(14,8))
    plt.plot(mse)
    plt.title('MSE of models')
    plt.legend(['ARMA', 'LSTM', 'GRU', 'RF', 'Zero return'])
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
    
def data_split(study_periods):
    train_ratio = 0.5
    valid_ratio = 0.25
    
    train_size = np.round(np.size(study_periods[0], 1) * train_ratio).astype(int)
    valid_size = np.round(np.size(study_periods[0], 1) * valid_ratio).astype(int)
    test_size = (np.size(study_periods[0], 1) - train_size - valid_size).astype(int)
    
    return train_size, valid_size, test_size

def append_periods(model_names, frequencies, frequencies_number_of_samples):
    study_periods_predictions = list()
    study_periods_returns = list()
    study_periods_dates = list()
    study_periods_number_of_study_periods = list()
    study_periods_study_periods = list()

    for frequency_index in range(5):
        print(f'Frequency: {frequencies[frequency_index]}')
        number_of_study_periods, study_periods, Data, dates = creating_study_periods(frequencies,\
                                                                                     frequencies_number_of_samples,\
                                                                                     frequency_index)
        train_size, valid_size, test_size = data_split(study_periods)
        
        predictions = np.zeros((len(model_names)+1, number_of_study_periods*test_size))
        
        for model_index in range(len(model_names)):
    #         print(model_names[model_index])
            predictions[model_index] = pd.read_csv('results/'+str(model_names[model_index])+\
                                                    '_predictions_frequency_'+str(frequencies[frequency_index])+\
                                                    '.csv', header=None).dropna(axis='columns').values.flatten()

        predictions[-1] = np.mean(predictions[:-1], axis=0)
        study_periods_predictions.append(predictions)
        study_periods_returns.append(study_periods[0,:,-test_size:].flatten())
        study_periods_dates.append(dates[-predictions.shape[1]:])
        study_periods_number_of_study_periods.append(number_of_study_periods)
        study_periods_study_periods.append(study_periods)
    return  study_periods_predictions, study_periods_returns, study_periods_dates,\
            study_periods_number_of_study_periods, study_periods_study_periods

def np_to_latex_table(data, name, calculate_mean=False, accuracy=2):
    if calculate_mean:
        table = np.zeros((data.shape[0]+1, data.shape[1]+1))
        table[:-1, :-1] = data
        table[-1, :-1] = np.mean(data, axis=0)
        table[:-1, -1] = np.mean(data, axis=1)
        table[-1, -1] = np.mean(np.mean(data))
    else:
        table = data
    
    if 0.1<np.mean(np.mean(table)):
        fmt = '%1.'+str(accuracy)+'f'
    else:
        fmt = '%2.'+str(accuracy)+'e'

#     print(table)
    np.savetxt(name, table, delimiter=' & ', fmt=fmt, newline=' \\\\\n')
    