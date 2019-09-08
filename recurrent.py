import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from bayes_opt import BayesianOptimization
import random
import time

# from libary import *

def divide_data(reshaped_x, reshaped_y, train_ratio, valid_ratio, look_back):
    train_size = np.round(np.size(reshaped_y, 0) * train_ratio).astype(int)
    valid_size = np.round(np.size(reshaped_y, 0) * valid_ratio).astype(int)
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

def train_recurrent_model(cell_type, number_of_study_periods, study_periods, train_ratio, valid_ratio):
    recurrent_start_time = time.time()
    init_points = 1
    n_iter = 1
    
    model_results = np.ones((number_of_study_periods,4))*np.Inf
    model_names = [None]*number_of_study_periods
    model_predictions = np.zeros((number_of_study_periods,study_periods.shape[2]))
    model_predictions[:] = np.nan
    
    def black_box_function(look_back, batch_size, optimizer, dropout, n_layers, first_layer, layer_decay):
        start_time = time.time()
        # Convert hyperparameters
        look_back = int(look_back)
        batch_size = 2**int(batch_size)
        n_epochs = batch_size
        optimizer = ['sgd','rmsprop','adam'][int(optimizer)]
        n_layers = int(n_layers)
        first_layer = int(first_layer)

        layers = []
        for k in range(0,n_layers):
            layers = np.append(layers, [first_layer*layer_decay**k+0.5]).astype(int)
        layers = np.clip(layers,1,None).tolist()

        # Reshape the data
        Reshaped = reshape(study_periods[0,period], look_back)

        # Get X and Y
        reshaped_x = Reshaped[:-1, :, :]
        reshaped_y = Reshaped[1:, -1, 0]

        # Divide in train, valid and test set
        train_x, train_y, valid_x, valid_y, test_x, test_y =\
            divide_data(reshaped_x, reshaped_y, train_ratio, valid_ratio, look_back)
            
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
        input_dim = (look_back, np.size(Reshaped,2))
        if cell_type == 'LSTM':
            model = LSTM_network(input_dim,layers,dropout)
        else:
            model = GRU_network(input_dim,layers,dropout)
        model.compile(loss='mse', optimizer=optimizer)

        # Print model summary
        #model.summary()

        # Train model
        # Fit network
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=8,\
                                      verbose=0, mode='auto', baseline=None, restore_best_weights=True)
        print(f'Time: {np.round((time.time()-start_time)/60,2)}')
        history = model.fit(train_norm_x, train_norm_y, epochs=n_epochs, batch_size=batch_size,\
                    validation_data=(valid_norm_x, valid_norm_y), verbose=0, shuffle=False, callbacks=[earlystopping])
        print(f'Time: {np.round((time.time()-start_time)/60,2)}')
        plt.plot(history.history['loss'],label='loss')
        plt.plot(history.history['val_loss'],label='val loss')
        plt.legend()
        plt.show()
        
        mse = np.mean(np.square((model.predict(valid_norm_x)*std+mean).flatten()-valid_y))

        if mse < model_results[period,1]:
            model_names[period] = NAME
            model_results[period, 0] = np.mean(np.square((model.predict(train_norm_x)*std+mean).flatten()-train_y))
            model_results[period, 1] = mse
            
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
            
            model_results[period, 2] = np.mean(np.square((model.predict(train_valid_norm_x)*std_tv+mean_tv).flatten()-train_valid_y))
            model_results[period, 3] = np.mean(np.square((model.predict(test_norm_tv_x)*std_tv+mean_tv).flatten()-test_y))
            model_predictions[period, -len(test_x):] = (model.predict(test_norm_tv_x)*std_tv+mean_tv)[:,0]

        # Clear model
        del model
        K.clear_session()
        print(f'Time: {np.round((time.time()-start_time)/60,2)}')
        return -mse
    
    for period in range(number_of_study_periods):
        print(f'Period: {period}')
        
        pbounds = {'look_back' : (1, 40),\
                    'batch_size' : (4, 10),\
                    'optimizer' : (0, 2),\
                    'dropout' : (0, 0.5),\
                    'n_layers' : (1, 4),\
                    'first_layer' : (1, 40),\
                    'layer_decay' : (0.3, 1)}

        optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds, random_state=None)
        
        start_time = time.time()
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        print(f'Time: {np.round((time.time()-start_time)/60,2)}')
       
    pd.DataFrame(model_names).to_csv('results/'+str(cell_type)+'_names_frequency_'+str(frequencies[frequency_index])+'.csv',\
                                         index=False, header=False)
    pd.DataFrame(model_results).to_csv('results/'+str(cell_type)+'_mse_frequency_'+str(frequencies[frequency_index])+'.csv',\
                                         index=False, header=False)
    pd.DataFrame(model_predictions).to_csv('results/'+str(cell_type)+'_predictions_frequency_'+str(frequencies[frequency_index])+'.csv',\
                                         index=False, header=False)
        
    print(f'{cell_type} trining time: {np.round((time.time()-recurrent_start_time)/60,2)}')        
    return model_names, model_results, model_predictions