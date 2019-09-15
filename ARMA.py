from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

def predict(coef, history):
    yhat = 0.0
    for i in range(1, len(coef)+1):
        yhat += coef[i-1] * history[-i]
    return yhat

def train_ARMA(number_of_study_periods, study_periods, train_ratio, valid_ratio,\
                                                             frequency_index, frequencies, frequencies_number_of_samples):
    ARMA_start_time = time.time()
    model_results = np.ones((number_of_study_periods,2))*np.Inf
    model_names = [None]*number_of_study_periods
    
    train_size = np.round(study_periods.shape[2] * train_ratio).astype(int)
    valid_size = np.round(study_periods.shape[2] * valid_ratio).astype(int)
    test_size = int(study_periods.shape[2]-train_size-valid_size)
    
    mse = np.zeros((number_of_study_periods,2))
    parameters = np.zeros((number_of_study_periods,2))
    predictions = np.zeros((number_of_study_periods,study_periods.shape[2]))
    predictions[:] = np.nan
    for period in range(number_of_study_periods):
        X = study_periods[0,period]
        train, test = X[:train_size+valid_size], X[train_size+valid_size:]
        
        mean = np.mean(train)
        std = np.std(train)
        train_norm, test_norm = (train-mean)/std, (test-mean)/std

        # fit model
        model = auto_arima(train_norm, exogenous=None, start_p=0, start_q=0, max_p=5, max_q=0, max_order=10, seasonal=False,\
                           stationary=True,  information_criterion='aic', alpha=0.05, test='kpss', stepwise=True, n_jobs=1,\
                           solver='nm', maxiter=1000, disp=0, suppress_warnings=True, error_action='ignore',\
                           return_valid_fits=False, out_of_sample_size=0, scoring='mae')
        mse[period,0] = np.mean(np.square(train-(model.predict_in_sample()*std+mean)))
        
        forecast = list()
        for t in range(len(test_norm)):
            yhat = model.predict(n_periods=1)[0]
            model.arima_res_.model.endog = np.append(model.arima_res_.model.endog, [test_norm[t]])
            forecast.append(yhat)
        
        forecast = np.array(forecast)*std+mean
        mse[period,1] = np.mean(np.square(forecast-test))
        predictions[period,-len(forecast):] = forecast
        parameters[period] = [int(model.order[0]), int(model.order[2])]
        
        print(f'Period: {period}, order: {parameters[period]}, mse: {mse[period]}')
    
    pd.DataFrame(parameters).to_csv('results/ARMA_names_frequency_'+str(frequencies[frequency_index])+'.csv',\
                                         index=False, header=False)
    pd.DataFrame(mse).to_csv('results/ARMA_mse_frequency_'+str(frequencies[frequency_index])+'.csv',\
                                         index=False, header=False)
    pd.DataFrame(predictions).to_csv('results/ARMA_predictions_frequency_'+str(frequencies[frequency_index])+'.csv',\
                                         index=False, header=False)
    
    print(f'ARMA training time: {np.round((time.time()-ARMA_start_time)/60,2)} minutes')
    return parameters, mse, predictions