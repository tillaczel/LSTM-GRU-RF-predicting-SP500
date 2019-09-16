# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from manipulate_data import *
from recurrent import *
from ARMA import *

def train(frequency_index, frequencies, frequencies_number_of_samples):

    print(f'Frequency: {frequencies[frequency_index]}')
    number_of_study_periods, study_periods, Data, dates = creating_study_periods(frequencies,\
                                                                                 frequencies_number_of_samples,\
                                                                                 frequency_index)
    # visualize_data(Data, Returns)

    print('ARMA')
    ARMA_parameters, ARMA_mse, ARMA_predictions = train_ARMA(number_of_study_periods, study_periods,\
                                                             frequency_index, frequencies, frequencies_number_of_samples)

    print('LSTM')
    LSTM_names, LSTM_mse, LSTM_predictions = train_recurrent_model('LSTM', number_of_study_periods ,study_periods,\
                                                             frequency_index, frequencies, frequencies_number_of_samples)
    print('GRU')
    GRU_names, GRU_mse, GRU_predictions = train_recurrent_model('GRU', number_of_study_periods, study_periods,\
                                                             frequency_index, frequencies, frequencies_number_of_samples)

    visualize_results((np.concatenate((np.reshape(ARMA_mse[:,-1], [number_of_study_periods,1]),\
                          np.reshape(LSTM_mse[:,-1], [number_of_study_periods,1]),\
                          np.reshape(GRU_mse[:,-1], [number_of_study_periods,1]),\
                          np.reshape(np.mean(np.square(study_periods[0,:,-ARMA_predictions.shape[1]:]),axis=1),\
                                                         [number_of_study_periods,1])), axis=1)))