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

def train_ARMA(number_of_study_periods,study_periods):
    model_results = np.ones((number_of_study_periods,2))*np.Inf
    model_names = [None]*number_of_study_periods
    
    train_size = int(study_periods.shape[2]*0.75)
    mse = np.zeros((number_of_study_periods,3,3))
    for period in range(number_of_study_periods):
        print(f'Period: {period}')
        X = study_periods[0,period]
        train, test = X[0:train_size], X[train_size:]
        
        
        for p in range(2):
            for q in range(2):
                history = [x for x in train]
                forecast = list()
                
                # fit model
                model = ARMA(history, order=(p,q))
                model_fit = model.fit(disp=0, maxiter=5000)#,start_params=np.zeros((p+q)), method='css'
                ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
                resid = model_fit.resid
                for t in range(len(test)):
                    yhat = predict(ar_coef, history) + predict(ma_coef, resid)
                    forecast.append(yhat)
                    history.append(test[t])

                mse[period,p,q] = np.mean(np.square(forecast-test[t]))
    return mse