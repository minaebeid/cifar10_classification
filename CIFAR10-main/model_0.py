from keras.models import Model
import tensorflow as tf
from keras.layers import Input, Conv1D, Dense, Flatten, Activation, BatchNormalization

def model(input_shape):
    input = Input(input_shape)
    X = Flatten()(input)
    X = Dense(units=64, activation='relu')(X)
    X = Dense(units=10, activation='softmax')(X)

    model = Model(inputs=input, outputs=X)
    return model