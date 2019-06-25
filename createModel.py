import tensorflow.keras as keras


def model_1d(n_dim):
    """create regression model with Dense layers"""

    model = keras.Sequential()
    model.add(keras.layers.Dense(480, input_dim=n_dim, kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(240, kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(120, kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(1, kernel_initializer='normal'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


if __name__ == "__main__":
    n_dim = 160
    model = model_1d(n_dim)
    model.summary()
