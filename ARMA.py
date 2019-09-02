from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARMA
import matplotlib.pyplot as plt
import numpy as np

def predict(coef, history):
    yhat = 0.0
    for i in range(1, len(coef)+1):
        yhat += coef[i-1] * history[-i]
    return yhat

def train_ARMA(number_of_study_periods, study_periods, train_ratio, valid_ratio):
    model_results = np.ones((number_of_study_periods,2))*np.Inf
    model_names = [None]*number_of_study_periods
    
    train_size = np.round(study_periods.shape[2] * train_ratio).astype(int)
    valid_size = np.round(study_periods.shape[2] * valid_ratio).astype(int)
    
    
    max_p, max_q = 5, 0
    mse_valid = np.zeros((number_of_study_periods, max_p+1, max_q+1))
    mse = np.zeros((number_of_study_periods))
    parameters = np.zeros((2, number_of_study_periods))
    for period in range(number_of_study_periods):
        X = study_periods[0,period]
        train, valid, test = X[0:train_size], X[train_size:train_size+valid_size], X[train_size+valid_size:]
        
        mean = np.mean(train)
        std = np.std(train)
        train, valid = (train-mean)/std, (valid-mean)/std
        
        for p in range(max_p+1):
            
            for q in range(max_q+1):
                history = [x for x in train]
#                 print('p:', p, 'q:', q)
                forecast = list()
                
                # fit model
                model = ARMA(history, order=(p,q))
                model_fit = model.fit(disp=0, solver='nm')#, method='css',maxiter=100, start_params=np.ones((p+q))
                ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
                resid = model_fit.resid
                
                for t in range(len(valid)):
                    yhat = predict(ar_coef, history) + predict(ma_coef, resid)
                    forecast.append(yhat)
                    history.append(valid[t])

                mse_valid[period,p,q] = np.mean(np.square(forecast-valid))
                
        history = [x for x in np.concatenate((train, valid))]
        forecast = list()
        
        if max_p ==0:
            if max_q == 0:
                p = 0
                q = 0
            else:
                p = 0
                q = np.argmax(mse_valid[period])
        else:
            if max_q == 0:
                p = np.argmax(mse_valid[period])
                q = 0
            else:
                p = np.argmax(mse_valid[period])[0]
                q = np.argmax(mse_valid[period])[1]
            
        
        # fit model
        model = ARMA(history, order=(p,q))
        model_fit = model.fit(disp=0, solver='nm', maxiter=1000)#, method='css', start_params=np.ones((p+q))
        ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
        resid = model_fit.resid

        for t in range(len(test)):
            yhat = predict(ar_coef, history) + predict(ma_coef, resid)
            forecast.append(yhat)
            history.append(test[t])
        mse[period] = np.mean(np.square(forecast-test))
        parameters[0,period] = p
        parameters[1,period] = q
        print(f'Period: {period}, p: {p}, q: {q}, mse: {mse[period]}')
    return mse, parameters