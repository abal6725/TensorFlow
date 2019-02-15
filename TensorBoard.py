from time import time

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import TensorBoard

model = Sequential()

model.add(Dense(10, input_shape=(784,)))
model.add(Activation('softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy')

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.fit(partial_x_train,
                    partial_y_train,
                    epochs=1000,
                    batch_size=1000,
                    validation_data=(x_val, y_val),
                    verbose=1, callbacks=[tensorboard])

