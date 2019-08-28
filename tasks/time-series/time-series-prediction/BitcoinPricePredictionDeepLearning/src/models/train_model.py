###Import packages for the creation of neuronal networks
import tensorflow as tf
import keras
from keras import backend as K

sess = tf.Session() 
keras.backend.set_session(sess)

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Input
from keras.optimizers import Adam, Nadam, SGD
from keras.activations import softplus
from keras.losses import categorical_crossentropy, logcosh

# fit an LSTM network to training data
def fit_lstm(x_train, y_train, x_val, y_val, params):

  batch_size = 1

  model = Sequential()

  model.add(LSTM(params['first_neuron'], batch_input_shape=(params['batch_size'], x_train.shape[1], x_train.shape[2]), return_sequences=True, stateful=True))
  #	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
  model.add(Dropout(params['dropout']))

  model.add(LSTM(params['second_neuron']))
  model.add(Dropout(params['dropout']))

  model.add(Dense(units=1))
  model.add(Activation(params['last_activation']))
  model.compile(loss='mean_squared_error', optimizer=params['optimizer'])
  #	for i in range(nb_epoch):

  out = model.fit(x_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=1, shuffle=False)
  model.reset_states()
  return out, model
