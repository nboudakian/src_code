import math
import numpy as np
import pandas_datareader as data_reader


# Sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Price format function
def stocks_price_format(n):
    if n < 0:
        return "- $ {0:2f}".format(abs(n))
    else:
        return "$ {0:2f}".format(abs(n))


# Dataset loader
def dataset_loader(stock_name):
    # Complete the dataset loader function
    dataset = data_reader.DataReader(stock_name, data_source="yahoo")

    start_date = str(dataset.index[0]).split()[0]
    end_date = str(dataset.index[-1]).split()[0]

    close = dataset['Close']

    return close


# State creator
def state_creator(data, time_step, window_size):
    starting_id = time_step - window_size + 1

    if starting_id >= 0:
        windowed_data = data[starting_id:time_step + 1]
    else:
        windowed_data = - starting_id * [data[0]] + list(data[0:time_step + 1])

    state = []
    for i in range(window_size - 1):
        state.append(sigmoid(windowed_data[i + 1] - windowed_data[i]))

    return np.array([state])
