from keras.models import Model
import tensorflow as tf
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Activation, BatchNormalization

def model(input_shape):
    input = Input(input_shape)

    X = Conv2D(filters=32, kernel_size=(3,3), padding='same')(input)
    X = Conv2D(filters=32, kernel_size=(3,3), padding='same')(X) 
    X = Activation(activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = Conv2D(filters=64, kernel_size=(3,3), padding='same')(X)
    X = Conv2D(filters=64, kernel_size=(3,3), padding='same')(X) 
    X = Activation(activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = Flatten()(X)
    X = Dense(units=64, activation='relu')(X)
    X = Dense(units=32, activation='softmax')(X)

    model = Model(inputs=input, outputs=X)
    return model