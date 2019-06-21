import scipy.io
import numpy as np

data_path = 'Z:/13_deepPhi/'
files_train = ['conn_matrices.mat', 'network_matrices.mat', 'phismax.mat', 'phismeans.mat']

conn_matrices = network_matrices = phismax = phismeans = []

for file in files_train:
    vars()[file[:-4]] = scipy.io.loadmat(data_path + file)['mat']

phismax = phismax.tolist()[0]
phismeans = phismeans.tolist()[0]

import tensorflow as tf
import createModel

dim = conn_matrices.shape
conn_matrices = np.reshape(conn_matrices, (dim[0], dim[1] * dim[2]))
model = createModel.model_1d(conn_matrices.shape[1])


