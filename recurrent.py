import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from bayes_opt import BayesianOptimization
import random
import time

# from libary import *

def divide_data(reshaped_x, reshaped_y, train_ratio, valid_ratio, look_back):
    train_size = np.round((np.size(reshaped_y, 0)+look_back) * train_ratio-look_back).astype(int)
    valid_size = np.round((np.size(reshaped_y, 0)+look_back) * valid_ratio).astype(int)
    test_size = (np.size(reshaped_y, 0) - train_size - valid_size).astype(int)
    train_x = reshaped_x[:train_size, :, :]
    train_y = reshaped_y[:train_size]
    valid_x = reshaped_x[train_size:train_size + valid_size, :, :]
    valid_y = reshaped_y[train_size:train_size + valid_size]
    test_x = reshaped_x[train_size + valid_size:, :, :]
    test_y = reshaped_y[train_size + valid_size:]
    
#     train_prices = prices[look_back:look_back+train_size]
#     valid_prices = prices[look_back+train_size:look_back+train_size+valid_size]
#     test_prices = prices[look_back+train_size+valid_size:look_back+train_size+valid_size+test_size]
    return train_x, train_y, valid_x, valid_y, test_x, test_y#, train_prices, valid_prices, test_prices

def reshape(Returns, look_back):
    # Ensure all data is float
    values = Returns.astype('float32')
    # Reshape the data
    values = values.reshape(np.size(values, 0), 1, 1)
    reshaped = np.empty([np.size(values, 0)-look_back+1, 0, np.size(values, 2)])
    # Timesteps in order of time
    for i in range(1, look_back+1):
        reshaped = np.concatenate((reshaped, np.roll(values, look_back-i, axis=0)[look_back-1:, :, :]), axis=1)
    return reshaped

def GRU_network(input_dim,layers,dropout):
    model_input = Input(shape=input_dim)
    x = GRU(layers[0], input_shape=input_dim,\
            return_sequences=False if np.size(layers,0)==1 else True)(model_input)
    x = Dropout(dropout)(x)
    if np.size(layers,0) > 2:
        for i in range(1,np.size(layers,0)-1):
            x = GRU(layers[i], return_sequences=True)(x)
            x = Dropout(dropout)(x)
    if np.size(layers,0) > 1:
        x = GRU(layers[-1])(x)
        x = Dropout(dropout)(x)
    model_output = Dense(1)(x)
    return Model(inputs=model_input, outputs=model_output)
    
def LSTM_network(input_dim,layers,dropout):
    model_input = Input(shape=input_dim)
    x = LSTM(layers[0], input_shape=input_dim,\
            return_sequences=False if np.size(layers,0)==1 else True)(model_input)
    x = Dropout(dropout)(x)
    if np.size(layers,0) > 2:
        for i in range(1,np.size(layers,0)-1):
            x = LSTM(layers[i], return_sequences=True)(x)
            x = Dropout(dropout)(x)
    if np.size(layers,0) > 1:
        x = LSTM(layers[-1])(x)
        x = Dropout(dropout)(x)
    model_output = Dense(1)(x)
    return Model(inputs=model_input, outputs=model_output)

#cell_type,number_of_study_periods,study_periods,number_of_random_search, train_ratio, valid_ratio, model_results, model_names, 

# class data:
#     def __init__(self,cell_type,number_of_study_periods,study_periods,number_of_random_search, train_ratio, valid_ratio):
#         self.cell_type = cell_type
#         self.study_periods = study_periods
#         self.train_ratio = train_ratio
#         self.valid_ratio = valid_ratio
        
#         self.model_results = np.ones((number_of_study_periods,3))*np.Inf
#         self.model_names = [None]*number_of_study_periods

def train_recurrent_model(cell_type, number_of_study_periods, study_periods, train_ratio, valid_ratio,\
                                                             frequency_index, frequencies, frequencies_number_of_samples):
    class recurrent_model():

        def __init__(self, cell_type, number_of_study_periods, study_periods, train_ratio, valid_ratio,\
                                                                 frequency_index, frequencies, frequencies_number_of_samples):
            self.cell_type = cell_type
            self.number_of_study_periods = number_of_study_periods
            self.study_periods = study_periods
            self.train_ratio = train_ratio
            self.valid_ratio = valid_ratio
            self.frequency_index = frequency_index
            self.frequencies = frequencies
            self.frequencies_number_of_samples = frequencies_number_of_samples


            self.recurrent_start_time = time.time()
            self.init_points = 2
            self.n_iter = 3

            self.model_results = np.ones((number_of_study_periods,4))*np.Inf
            self.model_names = [None]*number_of_study_periods
            self.model_predictions = np.zeros((number_of_study_periods,study_periods.shape[2]))
            self.model_predictions[:] = np.nan

        def black_box_function(self, look_back, batch_size, optimizer, dropout, n_layers, first_layer, layer_decay, learning_rate):
    #         start_time = time.time()
            # Convert hyperparameters
            look_back = int(look_back)
            batch_size = 2**int(batch_size)
            n_epochs = batch_size
            optimizer = ['sgd','rmsprop','adam'][int(optimizer)]
            n_layers = int(n_layers)
            first_layer = int(first_layer)
            learning_rate = np.exp(-learning_rate)

            layers = []
            for k in range(0,n_layers):
                layers = np.append(layers, [first_layer*layer_decay**k+0.5]).astype(int)
            layers = np.clip(layers,1,None).tolist()

            # Reshape the data
            Reshaped = reshape(self.study_periods[0,self.period], look_back)

            # Get X and Y
            reshaped_x = Reshaped[:-1, :, :]
            reshaped_y = Reshaped[1:, -1, 0]

            # Divide in train, valid and test set
            train_x, train_y, valid_x, valid_y, test_x, test_y =\
                divide_data(reshaped_x, reshaped_y, self.train_ratio, self.valid_ratio, look_back)

            mean = np.mean(np.append(train_x[0], train_y))
            std = np.std(np.append(train_x[0], train_y))

            train_norm_x, valid_norm_x, test_norm_x = (train_x-mean)/std, (valid_x-mean)/std, (test_x-mean)/std
            train_norm_y, valid_norm_y, test_norm_y = (train_y-mean)/std, (valid_y-mean)/std, (test_y-mean)/std

            train_valid_x = np.concatenate((train_x, valid_x))
            train_valid_y = np.concatenate((train_y, valid_y))

            mean_tv = np.mean(np.append(train_valid_x[0], train_valid_y))
            std_tv = np.std(np.append(train_valid_x[0], train_valid_y))

            train_valid_norm_x, test_norm_tv_x = (train_valid_x-mean_tv)/std_tv, (test_x-mean_tv)/std_tv
            train_valid_norm_y, test_norm_tv_y = (train_valid_y-mean_tv)/std_tv, (test_y-mean_tv)/std_tv



            # Name the model
            NAME = 'look_back-'+str(look_back)+\
                ', n_epochs-'+str(n_epochs)+\
                ', batch_size-'+str(batch_size)+\
                ', optimizer-'+optimizer+\
                ', layers-'+str(layers)+\
                ', dropout-'+str(dropout)
    #         print('Model name:', NAME)

            #Design model
            if optimizer == 'sgd':
                optimizer_k = optimizers.SGD(lr=learning_rate)
            elif optimizer == 'rmsprop':
                optimizer_k = optimizers.RMSprop(lr=learning_rate)
            elif optimizer == 'adam':
                optimizer_k = optimizers.Adam(lr=learning_rate)

            input_dim = (look_back, np.size(Reshaped,2))
            if cell_type == 'LSTM':
                model = LSTM_network(input_dim,layers,dropout)
            else:
                model = GRU_network(input_dim,layers,dropout)
            model.compile(loss='mae', optimizer=optimizer)

            # Print model summary
            #model.summary()

            # Train model
            # Fit network
            earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=8,\
                                          verbose=0, mode='auto', baseline=None, restore_best_weights=True)
    #         print(f'Time: {np.round((time.time()-start_time)/60,2)}')
            history = model.fit(train_norm_x, train_norm_y, epochs=n_epochs, batch_size=batch_size,\
                        validation_data=(valid_norm_x, valid_norm_y), verbose=0, shuffle=False, callbacks=[earlystopping])
    #         print(f'Time: {np.round((time.time()-start_time)/60,2)}')
    #         plt.plot(history.history['loss'],label='loss')
    #         plt.plot(history.history['val_loss'],label='val loss')
    #         plt.legend()
    #         plt.show()

            mse = np.mean(np.square((model.predict(valid_norm_x)*std+mean).flatten()-valid_y))

            if mse < self.model_results[self.period,1]:
                self.model_names[self.period] = NAME
                self.model_results[self.period, 0] = np.mean(np.square((model.predict(train_norm_x)*std+mean).flatten()-train_y))
                self.model_results[self.period, 1] = mse

                #Design model
                del model
                K.clear_session()
                input_dim = (look_back, np.size(Reshaped,2))
                if cell_type == 'LSTM':
                    model = LSTM_network(input_dim,layers,dropout)
                else:
                    model = GRU_network(input_dim,layers,dropout)
                model.compile(loss='mse', optimizer=optimizer)
                model.fit(train_valid_norm_x, train_valid_norm_y, epochs=n_epochs, batch_size=batch_size,\
                        validation_data=(test_norm_tv_x, test_norm_tv_y), verbose=0, shuffle=False, callbacks=[earlystopping])

                self.model_results[self.period, 2] = np.mean(np.square((model.predict(train_valid_norm_x)*std_tv+mean_tv)\
                                                                  .flatten()-train_valid_y))
                self.model_results[self.period, 3] = np.mean(np.square((model.predict(test_norm_tv_x)*std_tv+mean_tv).flatten()-test_y))
                self.model_predictions[self.period, -len(test_x):] = (model.predict(test_norm_tv_x)*std_tv+mean_tv)[:,0]

            # Clear model
            del model
            K.clear_session()
    #         print(f'Time: {np.round((time.time()-start_time)/60,2)}')
            return -mse

        def train(self):
            for self.period in range(self.number_of_study_periods):
                print(f'Period: {self.period}')

                pbounds = {'look_back' : (1, 40),\
                            'batch_size' : (4, 10),\
                            'optimizer' : (0, 2),\
                            'dropout' : (0, 0.5),\
                            'n_layers' : (1, 4),\
                            'first_layer' : (1, 40),\
                            'layer_decay' : (0.3, 1),\
                            'learning_rate' : (0, 15)}

                optimizer = BayesianOptimization(f=self.black_box_function, pbounds=pbounds, random_state=None)

                start_time = time.time()
                optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)
                print(f'Period time: {np.round((time.time()-start_time)/60,2)} minutes')

            pd.DataFrame(self.model_names).to_csv('results/'+str(self.cell_type)+'_names_frequency_'\
                                             +str(self.frequencies[self.frequency_index])+'.csv',index=False, header=False)
            pd.DataFrame(self.model_results).to_csv('results/'+str(self.cell_type)+'_mse_frequency_'\
                                               +str(self.frequencies[self.frequency_index])+'.csv',index=False, header=False)
            pd.DataFrame(self.model_predictions).to_csv('results/'+str(self.cell_type)+'_predictions_frequency_'\
                                                    +str(self.frequencies[self.frequency_index])+'.csv',index=False, header=False)

            print(f'{self.cell_type} training time: {np.round((time.time()-self.recurrent_start_time)/60,2)} minutes')        
    
    
    recurrent_model = recurrent_model(cell_type, number_of_study_periods, study_periods, train_ratio, valid_ratio,\
                                                             frequency_index, frequencies, frequencies_number_of_samples)
    recurrent_model.train()
    return recurrent_model.model_names, recurrent_model.model_results, recurrent_model.model_predictions