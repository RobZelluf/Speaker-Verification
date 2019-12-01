from keras . models import *
from keras . layers import *
from keras . constraints import *
from keras . engine import InputSpec


def GRU_NN(num_speak=100, dimensions=[30, 201]):
    input_shape = (dimensions[1], dimensions[0])
    model = Sequential()
    model.add(Bidirectional(CuDNNGRU(units=256, return_sequences=True), input_shape=input_shape))
    model.add(Dense(units=512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dense(units=256, activation="relu"))
    model.add(GlobalAveragePooling1D())
    model.add(BatchNormalization())
    model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))
    model.add(Dense(num_speak, activation="softmax"))
    return model
