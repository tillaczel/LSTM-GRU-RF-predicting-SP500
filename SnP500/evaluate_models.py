# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from arch.bootstrap.multiple_comparison import MCS  

from manipulate_data import *

import matplotlib.style
import matplotlib as mpl
mpl.style.use('seaborn-colorblind')

def change_font(SIZE):
    plt.rc('font', size=SIZE)          
    plt.rc('axes', titlesize=SIZE)     
    plt.rc('axes', labelsize=SIZE)    
    plt.rc('xtick', labelsize=SIZE)    
    plt.rc('ytick', labelsize=SIZE)    
    plt.rc('legend', fontsize=SIZE)    
    plt.rc('figure', titlesize=SIZE)
    plt.rc('font', family='serif')

def calculate_da_mse(model_names, frequencies, number_of_study_periods, study_periods):
    mse = np.zeros((5, len(model_names)+1, number_of_study_periods[0]))
    directional_accuracy = np.zeros((5, len(model_names)+1, number_of_study_periods[0]))
    for frequency_index in range(5):
        train_size, valid_size, test_size = data_split(study_periods[frequency_index])
        
        study_periods_direction = study_periods[frequency_index][0].copy()
        study_periods_direction[0<study_periods_direction] = 1
        study_periods_direction[0==study_periods_direction] = 0
        study_periods_direction[0>study_periods_direction] = -1
        
        prediction = np.zeros((len(model_names)+1, number_of_study_periods[frequency_index], test_size))
        for model_index in range(len(model_names)):
            mse[frequency_index, model_index] = pd.read_csv('results/'+str(model_names[model_index])+'_mse_frequency_'\
                                        +str(frequencies[frequency_index])+'.csv', header=None).values[:,-1]
            prediction[model_index] = pd.read_csv('results/'+str(model_names[model_index])+'_predictions_frequency_'\
                                        +str(frequencies[frequency_index])+'.csv', header=None).dropna(axis='columns').values
        prediction[-1] = np.mean(prediction[:-1], axis=0)
        mse[frequency_index, -1] = np.mean(np.square(prediction[-1]-study_periods[frequency_index][0][:, -test_size:]), axis=1)
        
        for model_index in range(len(model_names)+1):
            predictions_direction = prediction[model_index].copy()
            predictions_direction[0<predictions_direction] = 1
            predictions_direction[0==predictions_direction] = 0
            predictions_direction[0>predictions_direction] = -1
            
            for period in range(number_of_study_periods[frequency_index]):
                directions = np.multiply(study_periods_direction[period,-test_size:], predictions_direction[period])
                directions_right = np.sum(directions[directions==1])
                directional_accuracy[frequency_index, model_index, period] = directions_right/directions.shape[0]
        
    np_to_latex_table(np.transpose(np.mean(mse, axis=0)), 'tables/mse_freq.csv', calculate_mean=True)
    np_to_latex_table(np.mean(mse, axis=2), 'tables/mse_study_period.csv', calculate_mean=True)
    np_to_latex_table(np.transpose(np.mean(directional_accuracy, axis=0)), 'tables/directional_accuracy_freq.csv', calculate_mean=True)
    np_to_latex_table(np.mean(directional_accuracy, axis=2), 'tables/directional_accuracy_study_period.csv', calculate_mean=True)
    
    change_font(24)
    fig = plt.figure(figsize=(14,8))
    data = np.transpose(np.mean(mse, axis=0))
    plt.bar(np.arange(16)-0.32, data[:,0], 0.16, label='ARMA')
    plt.bar(np.arange(16)-0.16, data[:,1], 0.16, label='LSTM')
    plt.bar(np.arange(16), data[:,2], 0.16, label='GRU')
    plt.bar(np.arange(16)+0.16, data[:,3], 0.16, label='RF')
    plt.bar(np.arange(16)+0.32, data[:,4], 0.16, label='ENS')
    plt.xticks(np.arange(16), ['2009 2H', '2010 1H', '2010 2H', '2011 1H', '2011 2H', '2012 1H', \
                               '2012 2H', '2013 1H', '2013 2H', '2014 1H', '2014 2H', '2015 1H', \
                               '2015 2H', '2016 1H', '2016 2H', '2017 1H'], rotation=30, ha='right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
    plt.tight_layout()
    plt.savefig('figures/MSE_over_study_periods.png')
    plt.show()

    fig = plt.figure(figsize=(14,8))
    data = np.transpose(np.mean(directional_accuracy, axis=0))
    plt.bar(np.arange(16)-0.32, data[:,0], 0.16, label='ARMA')
    plt.bar(np.arange(16)-0.16, data[:,1], 0.16, label='LSTM')
    plt.bar(np.arange(16), data[:,2], 0.16, label='GRU')
    plt.bar(np.arange(16)+0.16, data[:,3], 0.16, label='RF')
    plt.bar(np.arange(16)+0.32, data[:,4], 0.16, label='ENS')
    plt.xticks(np.arange(16), ['2009 2H', '2010 1H', '2010 2H', '2011 1H', '2011 2H', '2012 1H', \
                               '2012 2H', '2013 1H', '2013 2H', '2014 1H', '2014 2H', '2015 1H', \
                               '2015 2H', '2016 1H', '2016 2H', '2017 1H'], rotation=30, ha='right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
    plt.tight_layout()
    plt.savefig('figures/Directional_accuracy_over_study_periods.png')
    plt.show()

    fig = plt.figure(figsize=(14,8))
    data = np.mean(mse, axis=2)
    plt.bar(np.arange(5)-0.32, data[:,0], 0.16, label='ARMA')
    plt.bar(np.arange(5)-0.16, data[:,1], 0.16, label='LSTM')
    plt.bar(np.arange(5), data[:,2], 0.16, label='GRU')
    plt.bar(np.arange(5)+0.16, data[:,3], 0.16, label='RF')
    plt.bar(np.arange(5)+0.32, data[:,4], 0.16, label='ENS')
    plt.xticks(np.arange(5), ['Day', '60 min', '15 min', '5 min', '1 min'])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
    plt.tight_layout()
    plt.savefig('figures/MSE_over_frequencies.png')
    plt.show()

    fig = plt.figure(figsize=(14,8))
    data = np.mean(directional_accuracy, axis=2)
    plt.bar(np.arange(5)-0.32, data[:,0], 0.16, label='ARMA')
    plt.bar(np.arange(5)-0.16, data[:,1], 0.16, label='LSTM')
    plt.bar(np.arange(5), data[:,2], 0.16, label='GRU')
    plt.bar(np.arange(5)+0.16, data[:,3], 0.16, label='RF')
    plt.bar(np.arange(5)+0.32, data[:,4], 0.16, label='ENS')
    plt.xticks(np.arange(5), ['Day', '60 min', '15 min', '5 min', '1 min'])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
    plt.tight_layout()
    plt.savefig('figures/Directional_accuracy_over_frequencies.png')
    plt.show()
    
    return mse, directional_accuracy

def calculate_trading_strategy(predictions, transaction_cost):
    strategies = list()
    for frequency_index in range(5):
        strategy = np.exp(predictions[frequency_index].copy())-1
#         strategy = np.concatenate((strategy, np.array([np.mean(strategy, axis=0)])), axis=0)
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

def create_logr(trading_strategy, returns, transaction_cost):
    logr = list()
    sum_logr = np.zeros((5, trading_strategy[0].shape[0]+1))
    for frequency_index in range(5):
        cost = -np.abs(np.diff(np.concatenate((np.zeros((trading_strategy[frequency_index].shape[0],1)),\
                               trading_strategy[frequency_index]), axis=1), axis=1))*transaction_cost
        
        logr.append(np.log(1+cost+np.multiply(trading_strategy[frequency_index], np.exp(returns[frequency_index])-1)))
        sum_logr[frequency_index,:-1] = np.sum(logr[-1], axis=1)
        sum_logr[frequency_index,-1] = np.sum(returns[frequency_index])
    np_to_latex_table(sum_logr, 'tables/sum_logr_'+str(transaction_cost).replace('.','')+'.csv')
    return logr

def vis_cum_logr(logr, returns, trading_strategy, frequencies, dates, number_of_study_periods, transaction_cost, model_names):
    cols = ['Cumulative logreturns', 'Cumulative trades']
    rows = ['Day', '60 minutes', '15 minutes', '5 minutes', '1 minute']
    
    change_font(12)
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(14, 18))

    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=24)

    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=90, fontsize=18)

        
    for frequency_index in range(5):
        dates_f = dates[frequency_index].dt.date.values
        date_index = (np.arange(number_of_study_periods[frequency_index]+1)/(number_of_study_periods[frequency_index])\
                      *dates_f.shape[0]).astype(int)
        date_index[-1] += -1

        l1, l2, l3, l4, l5 = axes[frequency_index, 0].plot(np.cumsum(np.transpose(logr[frequency_index]), axis=0))[:]
        l6 = axes[frequency_index, 0].plot(np.cumsum(returns[frequency_index]), linewidth=3)[0]
        for i in range(number_of_study_periods[frequency_index]+1):
            axes[frequency_index, 0].axvline(x=(i/(number_of_study_periods[frequency_index])*dates_f.shape[0]).astype(int),\
                                             linestyle='--', c='black', linewidth=1)
        plt.sca(axes[frequency_index, 0])
        if frequency_index==4:
            plt.xticks(date_index, dates_f[date_index], rotation=90)
        else:
            plt.xticks([],[])

        axes[frequency_index, 1].plot(np.cumsum(np.abs(np.diff(np.transpose(trading_strategy[frequency_index]), axis=0)), axis=0),\
                                      linewidth=3)
        for i in range(number_of_study_periods[frequency_index]+1):
            axes[frequency_index, 1].axvline(x=(i/(number_of_study_periods[frequency_index])*dates_f.shape[0]).astype(int),\
                                             linestyle='--', c='black', linewidth=1)
        plt.sca(axes[frequency_index, 1])
        if frequency_index==4:
            plt.xticks(date_index, dates_f[date_index], rotation=90)
        else:
            plt.xticks([],[])

    
    fig.tight_layout()
    legen_names = model_names.copy()
    legen_names.extend(['ENS', 'S&P'])
    leg = fig.legend([l1, l2, l3, l4, l5, l6], legen_names, loc='lower center', borderaxespad=0.1, ncol=6)
    for line in leg.get_lines():
        line.set_linewidth(10)
    plt.subplots_adjust(bottom=0.1)
    plt.savefig('figures/Cumulative_logreturns_'+str(transaction_cost).replace('.','')+'.png')

    plt.show()
        
def create_sharpe_ratio(logr, returns, transaction_cost, frequencies_number_of_samples, model_names, rf=0):
    logr = logr.copy()
    returns = returns.copy()
    
    sharpe_ratio = np.zeros((5, len(model_names)+2))
    modelr = logr.copy()
    
    for frequency_index in range(5):
        modelr[frequency_index] = np.exp(logr[frequency_index])-1
        returns[frequency_index] = np.exp(returns[frequency_index])-1
        
        sharpe_ratio[frequency_index, :-1] =\
                            (np.mean(modelr[frequency_index], axis=1)-rf)/(np.std(modelr[frequency_index], axis=1)+1e-8)
        sharpe_ratio[frequency_index, -1] = (np.mean(returns[frequency_index])-rf)/(np.std(returns[frequency_index])+1e-8)
        sharpe_ratio[frequency_index] = sharpe_ratio[frequency_index]*frequencies_number_of_samples[frequency_index]**0.5
    np_to_latex_table(sharpe_ratio, 'tables/sharpe_ratio'+str(transaction_cost).replace('.','')+'.csv')
    return sharpe_ratio

def create_sortino_ratio(logr, returns, transaction_cost, frequencies_number_of_samples, model_names, rf=0):
    logr = logr.copy()
    returns = returns.copy()
    
    sortino_ratio = np.zeros((5, len(model_names)+2))
    
    for frequency_index in range(5):
        m_r = np.exp(logr[frequency_index])-1
        sp_r = np.exp(returns[frequency_index])-1

        sortino_ratio[frequency_index, :-1] =\
                    (np.mean(m_r, axis=1)-rf)/\
                    np.sqrt(np.sum(np.square(np.multiply(m_r-np.mean(m_r), m_r<np.mean(m_r))), axis=1)/m_r.shape[1]+1e-8)
        sortino_ratio[frequency_index, -1] =\
                    (np.mean(sp_r)-rf)/\
                    np.sqrt(np.sum(np.square(np.multiply(sp_r-np.mean(sp_r), sp_r<np.mean(sp_r))))/sp_r.shape[0]+1e-8)
        sortino_ratio[frequency_index] = sortino_ratio[frequency_index]*frequencies_number_of_samples[frequency_index]**0.5
    np_to_latex_table(sortino_ratio, 'tables/sortino_ratio'+str(transaction_cost).replace('.','')+'.csv')
    return sortino_ratio
            
def calculate_MCS(predictions, returns, model_names):  
    MCS_values = np.zeros((5, len(model_names)+1))
    for frequency_index in range(5):
        losses = np.transpose(np.square(predictions[frequency_index]-returns[frequency_index]))
        mcs = MCS(losses, size=0.1)
        mcs.compute()
        MCS_values[frequency_index] = mcs.pvalues.sort_index(axis = 0).values.flatten()
    np_to_latex_table(MCS_values, 'tables/MCS.csv', accuracy=4)
    
def calculate_pred_corr(predictions, returns, model_names):
    change_font(36)
    corr_labels = model_names.copy()
    corr_labels.extend(['ENS', 'S&P'])
    for frequency_index in range(5):
        corr_matrix = np.corrcoef(np.concatenate((predictions[frequency_index], returns[frequency_index][np.newaxis, :]), axis=0))
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap='RdYlGn')
        plt.xticks(np.arange(len(corr_labels)), corr_labels)
        ax.xaxis.tick_top()
        plt.yticks(np.arange(len(corr_labels)), corr_labels, rotation=90, va="center")
        for i in range(len(corr_labels)):
            for j in range(len(corr_labels)):
                text = ax.text(j, i, np.round(corr_matrix[i, j],2), ha="center", va="center", color="black")
        ax.set_ylim(len(corr_labels)-0.5, -0.5)
        plt.savefig('figures/pred_corr_matrix_frequency_'+str(frequency_index)+'.png')
        plt.show()
        
def calculate_error_corr(predictions, returns, model_names):
    change_font(36)
    corr_labels = model_names.copy()
    corr_labels.extend(['ENS'])
    for frequency_index in range(5):
        corr_matrix = np.corrcoef(predictions[frequency_index]-returns[frequency_index])
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap='RdYlGn')
        plt.xticks(np.arange(len(corr_labels)), corr_labels)
        ax.xaxis.tick_top()
        plt.yticks(np.arange(len(corr_labels)), corr_labels, rotation=90, va="center")
        for i in range(len(corr_labels)):
            for j in range(len(corr_labels)):
                text = ax.text(j, i, np.round(corr_matrix[i, j],2), ha="center", va="center", color="black")
        ax.set_ylim(len(corr_labels)-0.5, -0.5)
        plt.savefig('figures/error_corr_matrix_frequency_'+str(frequency_index)+'.png')
        plt.show()
        
def calculate_pred_std(predictions, returns):
    pred_std = np.zeros((5, predictions[0].shape[0]+1))
    for frequency_index in range(5):
        pred_std[frequency_index][:-1] = np.std(predictions[frequency_index], axis=1)
        pred_std[frequency_index][-1] = np.std(returns[frequency_index])
    np_to_latex_table(pred_std, 'tables/pred_std.csv')
    return pred_std