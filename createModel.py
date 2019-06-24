import tensorflow.keras as keras


def model_1d():
    model = keras.Sequential()
    model.add(keras.layers.Dense(120, input_dim=160, kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(80, kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(1, kernel_initializer='normal'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


if __name__ == "__main__":
    dim = 160
    model = model_1d(dim)
    model.summary()
