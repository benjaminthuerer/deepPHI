import numpy as np
import createModel
import matplotlib.pyplot as plt
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns


def cross_validation(estimator, train_data, train_target, val_data, val_target):
    """10-fold cross-validation of KerasRegressor with mean-squared-error and R-squared for scoring"""

    kfold = KFold(n_splits=10, random_state=7)
    results_r = cross_val_score(estimator, train_data, train_target, cv=kfold, scoring="r2")
    print("Results: %.2f (%.2f) R2" % (results_r.mean(), results_r.std()))

    results_mse = cross_val_score(estimator, train_data, train_target, cv=kfold, scoring="neg_mean_squared_error")
    print("Results: %.2f (%.2f) MSE" % (results_mse.mean(), results_mse.std()))

    # fit again for plotting
    estimator.fit(train_data, train_target)
    predictions = estimator.predict(val_data)

    # use seaborn for plotting
    mse = mean_squared_error(val_target, predictions)
    r2 = r2_score(val_target, predictions)

    ax = sns.regplot(x=val_target, y=predictions)
    ax.set_title("MSE: %.2f, R2: %.2f" % (mse, r2))
    ax.set_xlabel("computed PHI max")
    ax.set_ylabel("predicted PHI max")
    plt.show()

    """old scatter plot"""
    # font = {'family': 'serif',
    #         'color':  'red',
    #         'weight': 'normal',
    #         'size': 16,
    #         }
    # plt.scatter(val_target, predictions)
    # plt.xlabel("PHI max values")
    # plt.ylabel("predicted PHI max")
    # plt.axis("equal")
    # plt.axis("square")
    # plt.xlim(-2, 12)
    # plt.ylim(-2, 12)
    # _ = plt.plot([-100, 100], [-100, 100], "r--")
    # plt.title("MSE: %.2f, R2: %.2f" % (mse, r2), fontdict=font)
    # plt.show()

    """old regressor without cross-validation"""
    # model = createModel.model_1d()
    # es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    # model.fit(train_data, train_target, batch_size=50, epochs=100, validation_split=0.2, callbacks=[es])


def train_model_1d(epochs, batch, train_data, val_data, train_target, val_target):
    """train KerasRegressor model to predict PHI with Dense model"""

    n_dim = train_data.shape[1]

    np.random.seed(7)
    estimator = KerasRegressor(build_fn=createModel.model_1d, n_dim=n_dim, epochs=epochs, batch_size=batch, verbose=2)

    cross_validation(estimator, train_data, train_target, val_data, val_target)


def train_model_cnn(epochs, batch, train_data, val_data, train_target, val_target):
    """train KerasRegressor model to predict PHI with 2D CNN"""

    n_dim = train_data.shape

    np.random.seed(7)
    estimator = KerasRegressor(build_fn=createModel.model_cnn, n_dim=n_dim, epochs=epochs, batch_size=batch,
                               verbose=2)

    cross_validation(estimator, train_data, train_target, val_data, val_target)
