import numpy as np

from keras import Sequential
from keras.layers import Dense, Dense, Flatten, Conv2D, MaxPool2D


def main():
    data = np.load("../data/raw/devanagari.npz")
    train_x, train_y, test_x, test_y = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# Concatenating train and test set into single train set
    X_train = np.concatenate([train_x, test_x])
    y_train = np.concatenate([train_y, test_y])

    X_train = X_train / 255

    X_train = np.expand_dims(X_train, axis=-1)
    y_train = y_train - 1

    model = Sequential()
    model.add(Conv2D(filters = 100, kernel_size=3, activation='relu', input_shape=(32, 32, 1)))
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Conv2D(filters = 50, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(46, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs = 5, batch_size = 128)

    # Save model for later use
    model.save('../out/nn-100.h5')


if __name__ == '__main__':
    main()
