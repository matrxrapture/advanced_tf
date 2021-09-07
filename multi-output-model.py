from numpy.core.fromnumeric import transpose
from numpy.lib import utils
from numpy.lib.financial import ipmt
from sklearn import model_selection
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow.python.autograph.utils.ag_logging import _output_to_stdout
from tensorflow.python.keras.engine.input_layer import InputLayer

from utills import format_output, norm, plot_diff


# prepare data

# Specify data URI
URI = './data/ENB2012_data.xlsx'

# Use pandas excel reader
df = pd.read_excel(URI)
df = df.sample(frac=1).reset_index(drop=True)

# split the data into train and test with 80/20 split
train, test = train_test_split(df, test_size=0.2)
train_stats = train.describe()


train_stats.pop('Y1')
train_stats.pop('Y2')
train_stats = train_stats.transpose()

train_Y = format_output(train)
test_Y = format_output(test)

# Normalise the training and test of data
norm_train_X = norm(train_stats, train)
norm_test_X = norm(train_stats, test)


# def building_model():

# Define layers
input_layer = Input(shape=(len(train .columns),))
first_dense = Dense(units='128', activation='relu')(input_layer)
second_dense = Dense(units='128', activation='relu')(first_dense)


# Y1 output will be fed ditrectly from second layer
y1_output = Dense(units='1', name='y1_output')(second_dense)
third_layer = Dense(units='64', activation='relu')(second_dense)

# Y2 output will be fed ditrectly from third layer
y2_output = Dense(units='1', name='y2_output')(second_dense)

# defining model

model = Model(inputs=input_layer, outputs=[y1_output, y2_output])


print(model.summary())


# Configure parameters
# Specify the optimizer, and compile the model with loss functions for both outputs
optimizer = tf.keras.optimizers.SGD(lr=0.001)
model.compile(optimizer=optimizer,
              loss={'y1_output': 'mse', 'y2_output': 'mse'},
              metrics={'y1_output': tf.keras.metrics.RootMeanSquaredError(),
                       'y2_output': tf.keras.metrics.RootMeanSquaredError()})


# Training the model
history = model.fit(norm_train_X, train_Y, epochs=500,
                    batch_size=10, validation_data=(norm_test_X, test_Y))

# evaluate model and plot metrics

loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(
    x=norm_test_X, y=test_Y)
print("Loss = {}, Y1_loss = {}, Y1_mse = {}, Y2_loss = {}, Y2_mse = {}".format(
    loss, Y1_loss, Y1_rmse, Y2_loss, Y2_rmse))

# Plot the loss and mse
Y_pred = model.predict(norm_test_X)
plot_diff(test_Y[0], Y_pred[0], title='Y1')
plot_diff(test_Y[1], Y_pred[1], title='Y2')
plot_metrics(metric_name='y1_output_root_mean_squared_error',
             title='Y1 RMSE', ylim=6)
plot_metrics(metric_name='y2_output_root_mean_squared_error',
             title='Y2 RMSE', ylim=7)
