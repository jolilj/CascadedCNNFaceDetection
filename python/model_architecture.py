from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Flatten
from keras.optimizers import SGD
import time

def setUpModel(windowSize):
    model = Sequential()
    model.add(Convolution2D(16, 3, 3, border_mode='valid', input_shape=(1, windowSize, windowSize)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1))

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True,clipnorm=100)
    print("======Compiling....======")
    start_time = time.time()
    model.compile(loss='mean_squared_error', optimizer=sgd)
    print("finished compiling in {0}".format(time.time()-start_time))

    return model

