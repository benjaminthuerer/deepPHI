import numpy as np
import createModel
import matplotlib.pyplot as plt
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def train_model(epochs, batch, train_data, val_data, train_target, val_target):
    """train keras Regressor model to predict PHI"""

    n_dim = train_data.shape[1]
    # model = createModel.model_1d(n_dim)

    np.random.seed(7)
    estimator = KerasRegressor(build_fn=createModel.model_1d, n_dim=n_dim, epochs=epochs, batch_size=batch, verbose=2)

    # 10-fold cross-validation
    kfold = KFold(n_splits=10, random_state=7)
    resultsR = cross_val_score(estimator, train_data, train_target, cv=kfold, scoring="r2")
    print("Results: %.2f (%.2f) R2" % (resultsR.mean(), resultsR.std()))

    resultsMSE = cross_val_score(estimator, train_data, train_target, cv=kfold, scoring="neg_mean_squared_error")
    print("Results: %.2f (%.2f) MSE" % (resultsMSE.mean(), resultsMSE.std()))

    """fit again for plotting"""
    estimator.fit(train_data, train_target)
    predictions = estimator.predict(val_data)

    font = {'family': 'serif',
            'color':  'red',
            'weight': 'normal',
            'size': 16,
            }

    plt.scatter(val_target, predictions)
    plt.xlabel("PHI max values")
    plt.ylabel("predicted PHI max")
    plt.axis("equal")
    plt.axis("square")
    plt.xlim(-2, 12)
    plt.ylim(-2, 12)
    _ = plt.plot([-100, 100], [-100, 100], "r--")
    plt.title("MSE: %.2f, R2: %.2f" % (resultsMSE.mean(), resultsR.mean()), fontdict=font)
    plt.show()


    """old regressor without cross-validation"""
    # model = createModel.model_1d()
    # es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    # model.fit(train_data, train_target, batch_size=50, epochs=100, validation_split=0.2, callbacks=[es])
