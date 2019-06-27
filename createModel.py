import tensorflow.keras as keras


def model_1d(dim):
    """create regression model with Dense layers"""

    model = keras.Sequential()
    model.add(keras.layers.Dense(480, input_dim=dim, kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(240, kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(120, kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(1, kernel_initializer='normal'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

    # for multi GPU parallelization
    # model_parallel = keras.utils.multi_gpu_model(model, gpus=2)
    # model_parallel.compile(optimizer='adam', loss='mean_squared_error')
    #
    # return model_parallel


def model_cnn(dim):
    """create regression model with 2D CNN"""

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(8, 2, input_shape=(dim[1], dim[2], 1), kernel_initializer='normal',
                                  activation='relu', data_format='channels_last'))
    model.add(keras.layers.MaxPooling2D(pool_size=1))

    model.add(keras.layers.Conv2D(12, 2, kernel_initializer='normal', activation='relu', data_format='channels_last'))
    model.add(keras.layers.MaxPooling2D(pool_size=1))

    model.add(keras.layers.Conv2D(16, 2, kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=1))

    model.add(keras.layers.Conv2D(20, 2, kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=1))

    # Conv1D instead? --> check for dimensions
    # model.add(keras.layers.Conv2D(16, 2, kernel_initializer='normal', activation='relu'))
    # model.add(keras.layers.MaxPooling2D(pool_size=1))

    model.add(keras.layers.Flatten())

    # Additional Dense layers worsened results (R2 down to .54)

    model.add(keras.layers.Dense(1, kernel_initializer='normal'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

    # for multi GPU parallelization
    # model_parallel = keras.utils.multi_gpu_model(model, gpus=2)
    # model_parallel.compile(optimizer='adam', loss='mean_squared_error')
    #
    # return model_parallel


def model_categorical(dim):
    """create categorical model to predict Phi values for 0, 0<PHI<1, 1, 1<PHI"""

    model = keras.Sequential()
    model.add(keras.layers.Dense(480, input_dim=dim, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(240, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(120, activation='relu'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    n_dim = 160
    model = model_1d(n_dim)
    model.summary()

    n_dim = [3000, 32, 5, 1]
    modelCNN = model_cnn(n_dim)
    modelCNN.summary()
