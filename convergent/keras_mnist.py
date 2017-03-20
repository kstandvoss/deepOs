import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

import experiment


def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    return (X_train, Y_train), (X_test, Y_test)


def build_model():
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
                                  # of the layer above. Here, with a "rectified linear unit",
                                  # we clamp all values below 0 to 0.

    model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax')) # This special "softmax" activation among other things,
                                     # ensures the output is a valid probaility distribution, that is
                                     # that its values are all non-negative and sum to 1.
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model



if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_mnist(cnn)


    model = build_model()
    model.fit(X_train, y_train,
        batch_size=128, epochs=4, verbose=1,
        validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)

    print('score', score)
