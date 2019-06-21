import tensorflow.keras as keras

def model_1d(dim):
    model = keras.Sequential()
    model.add(keras.layers.Dense(20, input_dim=dim, activation='relu'))
    model.add(keras.layers.BatchNormalization(0.2))
    model.add(keras.layers.Dense(14, activation='relu'))
    model.add(keras.layers.BatchNormalization(0.2))

    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_absolute_percentage_error')

