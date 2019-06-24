import scipy.io
import numpy as np
import createModel
import matplotlib.pyplot as plt
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

data_path = 'Z:/13_deepPhi/'
files_train = ['conn_matrices.mat', 'network_matrices.mat', 'phismax.mat', 'phismeans.mat']

conn_matrices = network_matrices = phismax = phismeans = []

for file in files_train:
    vars()[file[:-4]] = scipy.io.loadmat(data_path + file)['mat']

phismax = phismax.tolist()[0]
phismeans = phismeans.tolist()[0]

dim = conn_matrices.shape
conn_matrices = np.reshape(conn_matrices, (dim[0], dim[1] * dim[2]))
conn_matrices = conn_matrices[:phismax.__len__()]

"""Keras Regressor"""
np.random.seed(7)
estimator = KerasRegressor(build_fn=createModel.model_1d, epochs=50, batch_size=20, verbose=0)

# 10-fold cross-validation
kfold = KFold(n_splits=10, random_state=7)
resultsR = cross_val_score(estimator, conn_matrices, phismax, cv=kfold, scoring="r2")
print("Results: %.2f (%.2f) R2" % (resultsR.mean(), resultsR.std()))

resultsMSE = cross_val_score(estimator, conn_matrices, phismax, cv=kfold, scoring="neg_mean_squared_error")
print("Results: %.2f (%.2f) MSE" % (resultsMSE.mean(), resultsMSE.std()))

"""fit again for plotting"""
estimator.fit(conn_matrices, phismax)
predictions = estimator.predict(conn_matrices)

font = {'family': 'serif',
        'color':  'red',
        'weight': 'normal',
        'size': 16,
        }

plt.scatter(phismax, predictions)
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
# model.fit(conn_matrices, phismax, batch_size=50, epochs=100, validation_split=0.2, callbacks=[es])
