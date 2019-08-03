import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import random
import time

from libary import *


def select_hyperparameter(hyperparameter):
    random_number = random.randint(0, np.size(hyperparameter,0)-1)
    parameter = hyperparameter[random_number]
    return parameter

def select_hyperparameters():
    n_epochs = 100
    look_back = select_hyperparameter(np.arange(1,41))
    batch_size = select_hyperparameter(2 ** np.arange(8,11))
    optimizer = select_hyperparameter(['sgd','rmsprop','adam'])
    dropout = select_hyperparameter(np.round(np.linspace(0, 0.8, num=10),decimals=3))
    n_layers = select_hyperparameter(np.arange(1,3))
    first_layer = select_hyperparameter(np.arange(1,21))
    layer_decay = select_hyperparameter(np.linspace(0.3, 1, num=10))
    
    # Convert hyperparameters
    layers = []
    for k in range(0,n_layers):
        layers = np.append(layers, [first_layer*layer_decay**k+0.5]).astype(int)
    layers = np.clip(layers,1,None).tolist()
    
    return n_epochs, look_back, batch_size, optimizer, dropout, layers

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

def GRU_network(input_dim,layers,dropout):
    model_input = Input(shape=input_dim)
    x = GRU(layers[0], input_shape=input_dim,\
            return_sequences=False if np.size(layers,0)==1 else True)(model_input)
    x = Dropout(dropout)(x)
    if np.size(layers,0) > 2:
        for k in range(1,np.size(layers,0)-1):
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
        for k in range(1,np.size(layers,0)-1):
            x = LSTM(layers[i], return_sequences=True)(x)
            x = Dropout(dropout)(x)
    if np.size(layers,0) > 1:
        x = LSTM(layers[-1])(x)
        x = Dropout(dropout)(x)
    model_output = Dense(1)(x)
    return Model(inputs=model_input, outputs=model_output)

def train_recurrent_model(cell_type,number_of_study_periods,study_periods,number_of_random_search):
    train_ratio = 0.5
    valid_ratio = 0.25

    model_results = np.ones((number_of_study_periods,3))*np.Inf
    model_names = [None]*number_of_study_periods

    for period in range(number_of_study_periods):
        print(f'Period: {period}')
        for i in range(0,number_of_random_search):
            #print(i)
            # Select random hyperparameters
            n_epochs, look_back, batch_size, optimizer, dropout, layers = select_hyperparameters()

            # Reshape the data
            Reshaped = reshape(study_periods[0,period], look_back)

            # Get X and Y
            reshaped_x = Reshaped[:-1, :, :]
            reshaped_y = Reshaped[1:, -1, 0]

            # Divide in train, valid and test set
            train_x, train_y, valid_x, valid_y, test_x, test_y =\
                divide_data(reshaped_x, reshaped_y, train_ratio, valid_ratio, look_back)

            # Name the model
            NAME = 'look_back-'+str(look_back)+\
                ', n_epochs-'+str(n_epochs)+\
                ', batch_size-'+str(batch_size)+\
                ', optimizer-'+optimizer+\
                ', layers-'+str(layers)+\
                ', dropout-'+str(dropout)
            #print('Model name:', NAME)
            earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=4,\
                                          verbose=0, mode='auto', baseline=None, restore_best_weights=False)


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
            start = time.time()
            # Fit network
            history = model.fit(train_x, train_y, epochs=n_epochs, batch_size=batch_size, validation_data=(valid_x, valid_y),\
                                verbose=0, shuffle=False, callbacks=[earlystopping])
            end = time.time()
    #         plt.plot(history.history['loss'],label='loss')
    #         plt.plot(history.history['val_loss'],label='val loss')
    #         plt.legend()
    #         plt.show()

            if np.mean(np.square(model.predict(valid_x)-valid_y)) < model_results[period,1]:
                model_names[period] = NAME
                model_results[period,0] = np.mean(np.square(model.predict(train_x)-train_y))
                model_results[period,1] = np.mean(np.square(model.predict(valid_x)-valid_y))
                model_results[period,2] = np.mean(np.square(model.predict(test_x)-test_y))
                model.save('models/'+str(period)+'-'+NAME)
#                 print(f'Train error {model_results[period,0]}')
#                 print(f'Valid error {model_results[period,1]}')
#                 print(f'Test error {model_results[period,2]}')

            del model

#             print("Training time:", round(end - start,0))

            
            # Clear model
            K.clear_session()
    return model_names, model_results