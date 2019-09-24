# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from manipulate_data import *

def calculate_da_mse(model_names, frequencies, number_of_study_periods, study_periods):
    mse = np.zeros((5, 4, number_of_study_periods[0]))
    directional_accuracy = np.zeros((5, 4, number_of_study_periods[0]))
    for frequency_index in range(5):
        train_size, valid_size, test_size = data_split(study_periods[frequency_index])
        
        study_periods_direction = study_periods[frequency_index][0].copy()
        study_periods_direction[0<study_periods_direction] = 1
        study_periods_direction[0==study_periods_direction] = 0
        study_periods_direction[0>study_periods_direction] = -1
        
        prediction = np.zeros((4, number_of_study_periods[frequency_index], test_size))
        for model_index in range(3):
            mse[frequency_index, model_index] = pd.read_csv('results/'+str(model_names[model_index])+'_mse_frequency_'\
                                        +str(frequencies[frequency_index])+'.csv', header=None).values[:,-1]
            prediction[model_index] = pd.read_csv('results/'+str(model_names[model_index])+'_predictions_frequency_'\
                                        +str(frequencies[frequency_index])+'.csv', header=None).dropna(axis='columns').values
        prediction[-1] = np.mean(prediction[:-1], axis=0)
        mse[frequency_index, -1] = np.mean(np.square(prediction[-1]-study_periods[frequency_index][0][:, -test_size:]), axis=1)
        
        for model_index in range(4):
            predictions_direction = prediction[model_index].copy()
            predictions_direction[0<predictions_direction] = 1
            predictions_direction[0==predictions_direction] = 0
            predictions_direction[0>predictions_direction] = -1
            
            for period in range(number_of_study_periods[frequency_index]):
                directions = np.multiply(study_periods_direction[period,-test_size:], predictions_direction[period])
                directions_right = np.sum(directions[directions==1])
                directional_accuracy[frequency_index, model_index, period] = directions_right/directions.shape[0]
        
    np_to_latex_table(np.transpose(np.mean(mse, axis=0)), 'tables/mse_freq.csv')
    np_to_latex_table(np.mean(mse, axis=2), 'tables/mse_study_period.csv')
    np_to_latex_table(np.transpose(np.mean(directional_accuracy, axis=0)), 'tables/directional_accuracy_freq.csv')
    np_to_latex_table(np.mean(directional_accuracy, axis=2), 'tables/directional_accuracy_study_period.csv')
    
    fig = plt.figure(figsize=(14,8))
    data = np.transpose(np.mean(mse, axis=0))
    plt.bar(np.arange(16)-0.3, data[:,0], 0.2, label='ARMA')
    plt.bar(np.arange(16)-0.1, data[:,1], 0.2, label='LSTM')
    plt.bar(np.arange(16)+0.1, data[:,2], 0.2, label='GRU')
    plt.bar(np.arange(16)+0.3, data[:,3], 0.2, label='Ensemble')
    plt.xticks(np.arange(16), ['2009 second', '2010 first', '2010 second', '2011 first', '2011 second', '2012 first', \
                               '2012 second', '2013 first', '2013 second', '2014 first', '2014 second', '2015 first', \
                               '2015 second', '2016 first', '2016 second', '2017 first'], rotation='vertical')
    plt.legend()
    plt.title('MSE over the study periods')
    plt.show()

    fig = plt.figure(figsize=(14,8))
    data = np.transpose(np.mean(directional_accuracy, axis=0))
    plt.bar(np.arange(16)-0.3, data[:,0], 0.2, label='ARMA')
    plt.bar(np.arange(16)-0.1, data[:,1], 0.2, label='LSTM')
    plt.bar(np.arange(16)+0.1, data[:,2], 0.2, label='GRU')
    plt.bar(np.arange(16)+0.3, data[:,3], 0.2, label='Ensemble')
    plt.xticks(np.arange(16), ['2009 second', '2010 first', '2010 second', '2011 first', '2011 second', '2012 first', \
                               '2012 second', '2013 first', '2013 second', '2014 first', '2014 second', '2015 first', \
                               '2015 second', '2016 first', '2016 second', '2017 first'], rotation='vertical')
    plt.legend()
    plt.title('Directional accuracy over the study periods')
    plt.show()

    fig = plt.figure(figsize=(14,8))
    data = np.mean(mse, axis=2)
    plt.bar(np.arange(5)-0.3, data[:,0], 0.2, label='ARMA')
    plt.bar(np.arange(5)-0.1, data[:,1], 0.2, label='LSTM')
    plt.bar(np.arange(5)+0.1, data[:,2], 0.2, label='GRU')
    plt.bar(np.arange(5)+0.3, data[:,3], 0.2, label='Ensemble')
    plt.xticks(np.arange(5), ['Day', '60 min', '15 min', '5 min', '1 min'])
    plt.legend()
    plt.title('MSE over frequencies')
    plt.show()

    fig = plt.figure(figsize=(14,8))
    data = np.mean(directional_accuracy, axis=2)
    plt.bar(np.arange(5)-0.3, data[:,0], 0.2, label='ARMA')
    plt.bar(np.arange(5)-0.1, data[:,1], 0.2, label='LSTM')
    plt.bar(np.arange(5)+0.1, data[:,2], 0.2, label='GRU')
    plt.bar(np.arange(5)+0.3, data[:,3], 0.2, label='Ensemble')
    plt.xticks(np.arange(5), ['Day', '60 min', '15 min', '5 min', '1 min'])
    plt.legend()
    plt.title('Directional accuracy over frequencies')
    plt.show()
    
    return mse, directional_accuracy

def calculate_trading_strategy(predictions, transaction_cost):
    strategies = list()
    for frequency_index in range(5):
        strategy = np.exp(predictions[frequency_index].copy())-1
        strategy = np.concatenate((strategy, np.array([np.mean(strategy, axis=0)])), axis=0)
        strategy[transaction_cost<strategy] = 1
        strategy[-transaction_cost>strategy] = -1
        strategy[:,0][np.where(np.logical_and(strategy[:,0]<transaction_cost,\
                                                              strategy[:,0]>-transaction_cost))] = 0
        for i in range(1, strategy.shape[1]):
            strategy[:,i][np.where(np.logical_and(strategy[:,i]<transaction_cost,\
                                                                  strategy[:,i]>-transaction_cost))] =\
                            strategy[:,i-1][np.where(np.logical_and(strategy[:,i]<transaction_cost,\
                                                                                  strategy[:,i]>-transaction_cost))]
           
        strategies.append(strategy)
    return strategies

def create_cum_logr(trading_strategy, returns, transaction_cost):
    cum_logr = list()
    for frequency_index in range(5):
        cost = -np.abs(np.diff(np.concatenate((np.zeros((trading_strategy[frequency_index].shape[0],1)),\
                               trading_strategy[frequency_index]), axis=1), axis=1))*transaction_cost
        
        cum_logr.append(np.log(1+cost+np.multiply(trading_strategy[frequency_index], np.exp(returns[frequency_index])-1)))
    return cum_logr

def vis_cum_logr(cum_logr, returns, trading_strategy, frequencies, dates, number_of_study_periods):
    fig = plt.figure(figsize=(14,40))
    for frequency_index in range(5):
        plt.subplot(5, 1, frequency_index+1)
        plt.plot(np.cumsum(np.transpose(cum_logr[frequency_index]), axis=0))
        plt.plot(np.cumsum(returns[frequency_index]),  linewidth=3)
        dates_f = dates[frequency_index].dt.date.values
        date_index = (np.arange(number_of_study_periods[frequency_index]+1)/(number_of_study_periods[frequency_index])\
                      *dates_f.shape[0]).astype(int)
        date_index[-1] += -1
        plt.xticks(date_index, dates_f[date_index], rotation='vertical')
        plt.ylabel('Cumulative logreturn')
        for i in range(number_of_study_periods[frequency_index]+1):
            plt.axvline(x=(i/(number_of_study_periods[frequency_index])*dates_f.shape[0]).astype(int), linestyle='-', c='black')
        plt.title(f'Cumulative logreturn at frequency {frequencies[frequency_index]}')
        plt.legend(['ARMA', 'LSTM', 'GRU', 'Ensemble', 'S&P500'], loc='upper left')
    plt.suptitle('Cumulative logreturns', y=1.07, fontsize=24)
    plt.subplots_adjust(top = 1.05, bottom=0.01)
    plt.show()
    
    for frequency_index in range(5):
        fig = plt.figure(figsize=(14,4))
        plt.plot(np.transpose(trading_strategy[frequency_index]))
        dates_f = dates[frequency_index].dt.date.values
        date_index = (np.arange(number_of_study_periods[frequency_index]+1)/(number_of_study_periods[frequency_index])\
                      *dates_f.shape[0]-1).astype(int)
        date_index[-1] += -1
        plt.xticks(date_index, dates_f[date_index], rotation='vertical')
        plt.ylabel('Position')
        plt.title(f'Position at frequency {frequencies[frequency_index]}')
        plt.legend(['ARMA', 'LSTM', 'GRU', 'Ensemble'], loc='upper left')
        plt.show()
        
def create_shapre_ratio(cum_logr, returns):
    cum_logr = cum_logr.copy()
    returns = returns.copy()
    
    shapre_ratio = np.zeros((5, 5))
    for frequency_index in range(5):
        cum_logr[frequency_index] = np.exp(cum_logr[frequency_index])-1
        returns[frequency_index] = np.exp(returns[frequency_index])-1
        
        shapre_ratio[frequency_index, 0:-1] =\
                            (np.mean(cum_logr[frequency_index], axis=1)-0)/(np.std(cum_logr[frequency_index], axis=1)+1e-8)
        shapre_ratio[frequency_index, -1] = (np.mean(returns[frequency_index])-0)/(np.std(returns[frequency_index])+1e-8)
        print(shapre_ratio[frequency_index])
    return shapre_ratio
            