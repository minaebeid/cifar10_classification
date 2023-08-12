from keras.models import Model
import tensorflow as tf
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Activation, BatchNormalization

def model(input_shape):
    input = Input(input_shape)

    X = Conv2D(filters=32, kernel_size=(3,3), padding='same')(input)
    X = Conv2D(filters=32, kernel_size=(3,3), padding='same')(X) 
    X = BatchNormalization(axis=3)(X)
    X = Activation(activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(rate=.2)(X)

    X = Conv2D(filters=64, kernel_size=(3,3), padding='same')(X)
    X = Conv2D(filters=64, kernel_size=(3,3), padding='same')(X)
    X = BatchNormalization(axis=3)(X) 
    X = Activation(activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(rate=.2)(X)

    X = Conv2D(filters=128, kernel_size=(3,3), padding='same')(X)
    X = Conv2D(filters=128, kernel_size=(3,3), padding='same')(X)
    X = BatchNormalization(axis=3)(X) 
    X = Activation(activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(rate=.2)(X)

    X = Conv2D(filters=256, kernel_size=(3,3), padding='same')(X)
    X = Conv2D(filters=256, kernel_size=(3,3), padding='same')(X)
    X = BatchNormalization(axis=3)(X) 
    X = Activation(activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(rate=.2)(X)

    X = Flatten()(X)
    X = Dense(units=512, activation='relu')(X)
    X = Dropout(rate=.6)(X)
    X = Dense(units=256, activation='relu')(X)
    X = Dropout(rate=.5)(X)
    X = Dense(units=10, activation='softmax')(X)

    model = Model(inputs=input, outputs=X)
    return model