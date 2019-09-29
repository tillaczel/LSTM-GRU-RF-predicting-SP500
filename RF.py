import numpy as np
import time
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestRegressor

from manipulate_data import *

def divide_data(reshaped_x, reshaped_y, look_back, study_periods):
    train_size, valid_size, test_size = data_split(study_periods)
    train_size -= look_back
    train_x = reshaped_x[:train_size, :]
    train_y = reshaped_y[:train_size]
    valid_x = reshaped_x[train_size:train_size + valid_size, :]
    valid_y = reshaped_y[train_size:train_size + valid_size]
    test_x = reshaped_x[train_size + valid_size:, :]
    test_y = reshaped_y[train_size + valid_size:]
    
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def reshape(Returns, look_back):
    # Ensure all data is float
    values = Returns.astype('float32')
    # Reshape the data
    values = values.reshape(np.size(values, 0), 1)
    reshaped = np.empty([np.size(values, 0)-look_back+1, 0])
    # Timesteps in order of time
    for i in range(1, look_back+1):
        reshaped = np.concatenate((reshaped, np.roll(values, look_back-i, axis=0)[look_back-1:, :]), axis=1)
    return reshaped


def train_RF(number_of_study_periods, study_periods, frequency_index, frequencies, frequencies_number_of_samples):
    class RF_model():
        
        def __init__(self, number_of_study_periods, study_periods, frequency_index, frequencies, frequencies_number_of_samples):
            self.number_of_study_periods = number_of_study_periods
            self.study_periods = study_periods
            self.frequency_index = frequency_index
            self.frequencies = frequencies
            self.frequencies_number_of_samples = frequencies_number_of_samples

            self.RF_start_time = time.time()
            self.init_points = 1
            self.n_iter = 2

            self.model_results = np.ones((number_of_study_periods,4))*np.Inf
            self.model_names = [None]*number_of_study_periods
            self.model_predictions = np.zeros((number_of_study_periods,study_periods.shape[2]))
            self.model_predictions[:] = np.nan
        
        
        def black_box_function(self, look_back):
            look_back = int(look_back)

            # Reshape the data
            Reshaped = reshape(self.study_periods[0,self.period], look_back)

            # Get X and Y
            reshaped_x = Reshaped[:-1, :]
            reshaped_y = Reshaped[1:, -1]

            # Divide in train, valid and test set
            train_x, train_y, valid_x, valid_y, test_x, test_y =\
                divide_data(reshaped_x, reshaped_y, look_back, self.study_periods)

            train_valid_x = np.concatenate((train_x, valid_x))
            train_valid_y = np.concatenate((train_y, valid_y))

            # Name the model
            NAME = 'look_back-'+str(look_back)

            #Design model
            model = RandomForestRegressor(n_estimators=1024, n_jobs=-2, min_samples_split=0.001, max_features=1/3) 

            # Fit network
            model.fit(train_x, train_y)

            mse = np.mean(np.square(model.predict(valid_x).flatten()-valid_y))

            if mse < self.model_results[self.period,1]:
                self.model_names[self.period] = NAME
                self.model_results[self.period, 0] = np.mean(np.square(model.predict(train_x).flatten()-train_y))
                self.model_results[self.period, 1] = mse

                #Design model
                model = RandomForestRegressor(n_estimators=1024, n_jobs=-2, min_samples_split=0.001, max_features=1/3)
                
                # Fit network
                model.fit(train_valid_x, train_valid_y)

                self.model_results[self.period, 2] = np.mean(np.square(model.predict(train_valid_x).flatten()-train_valid_y))
                self.model_results[self.period, 3] = np.mean(np.square(model.predict(test_x).flatten()-test_y))
                self.model_predictions[self.period, -len(test_x):] = model.predict(test_x)
                
            return -mse
                
        def train(self):
            for self.period in range(self.number_of_study_periods):
                print(f'Period: {self.period}')

                pbounds = {'look_back' : (1, 40)}

                optimizer = BayesianOptimization(f=self.black_box_function, pbounds=pbounds, random_state=None)

                start_time = time.time()
                optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)
                print(f'Period time: {np.round((time.time()-start_time)/60,2)} minutes')

            pd.DataFrame(self.model_names).to_csv('results/RF_names_frequency_'\
                                             +str(self.frequencies[self.frequency_index])+'.csv',index=False, header=False)
            pd.DataFrame(self.model_results).to_csv('results/RF_mse_frequency_'\
                                               +str(self.frequencies[self.frequency_index])+'.csv',index=False, header=False)
            pd.DataFrame(self.model_predictions).to_csv('results/RF_predictions_frequency_'\
                                                +str(self.frequencies[self.frequency_index])+'.csv',index=False, header=False)

            print(f'RF training time: {np.round((time.time()-self.RF_start_time)/60,2)} minutes')        
    
    
    RF_model = RF_model(number_of_study_periods, study_periods, frequency_index, frequencies, frequencies_number_of_samples)
    RF_model.train()
    return RF_model.model_names, RF_model.model_results, RF_model.model_predictions