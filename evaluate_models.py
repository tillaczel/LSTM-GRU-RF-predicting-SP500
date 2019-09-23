# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from manipulate_data import *

def calculate_directional_accuracy(model_names, frequencies, frequencies_number_of_samples):
    
    
    study_periods_directional_accuracy = list()
    
    for frequency_index in range(5):
        print(f'Frequency: {frequencies[frequency_index]}')
        number_of_study_periods, study_periods, Data, dates = creating_study_periods(frequencies,\
                                                                                     frequencies_number_of_samples,\
                                                                                     frequency_index)
        train_size, valid_size, test_size = data_split(study_periods)
        
        study_periods_direction = study_periods[0].copy()

        study_periods_direction[0<study_periods_direction] = 1
        study_periods_direction[0==study_periods_direction] = 0
        study_periods_direction[0>study_periods_direction] = -1

        directional_accuracy = np.zeros((number_of_study_periods, 3))
        for model_index in range(3):
    #         print(model_names[model_index])
            predictions = pd.read_csv('results/'+str(model_names[model_index])+'_predictions_frequency_'\
                                        +str(frequencies[frequency_index])+'.csv', header=None).dropna(axis='columns').values
            
            predictions_direction = predictions.copy()

            predictions_direction[0<predictions_direction] = 1
            predictions_direction[0==predictions_direction] = 0
            predictions_direction[0>predictions_direction] = -1
            for period in range(number_of_study_periods):
                directions = np.multiply(study_periods_direction[period,-test_size:], predictions_direction[period])
                directions_right = np.sum(directions[directions==1])
                directional_accuracy[period,model_index] = directions_right/directions.shape[0]
    #             print(f'Directional accuracy in period {period}: {directional_accuracy_f[period,model_index]}')


        study_periods_directional_accuracy.append(directional_accuracy)

    return np.array(study_periods_directional_accuracy)

def vis_directional_accuracy(directional_accuracy, frequencies, model_names):
    for frequency_index in range(5):
        fig = plt.figure(figsize=(14,8))
        plt.plot(directional_accuracy[frequency_index])
        plt.title(f'Directional accuracy at frequency: {frequencies[frequency_index]}')
        plt.legend(model_names)
        plt.show()

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
    fig = plt.figure(figsize=(14,30))
    for frequency_index in range(5):
        plt.subplot(5, 1, frequency_index+1)
        plt.plot(np.cumsum(np.transpose(cum_logr[frequency_index]), axis=0))
        plt.plot(np.cumsum(returns[frequency_index]),  linewidth=3)
        dates_f = dates[frequency_index].dt.date.values
        date_index = (np.arange(number_of_study_periods[frequency_index]+1)/(number_of_study_periods[frequency_index])\
                      *dates_f.shape[0]-1).astype(int)
        plt.xticks(date_index, dates_f[date_index], rotation='vertical')
        plt.ylabel('Cumulative logreturn')
        for i in range(number_of_study_periods[frequency_index]+1):
            plt.axvline(x=(i/(number_of_study_periods[frequency_index])*dates_f.shape[0]).astype(int), linestyle='-', c='black')
        plt.title(f'Cumulative logreturn at frequency {frequencies[frequency_index]}')
        plt.legend(['ARMA', 'LSTM', 'GRU', 'Ensemble', 'S&P500'], loc='upper left')
    plt.suptitle('Cumulative logreturns', y=1.02, fontsize=24)
    plt.subplots_adjust(top = 1.05, bottom=0.01)
    plt.show()
    
    for frequency_index in range(5):
        fig = plt.figure(figsize=(14,4))
        plt.plot(np.transpose(trading_strategy[frequency_index]))
        dates_f = dates[frequency_index].dt.date.values
        date_index = (np.arange(number_of_study_periods[frequency_index]+1)/(number_of_study_periods[frequency_index])\
                      *dates_f.shape[0]-1).astype(int)
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