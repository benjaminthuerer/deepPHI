import scipy.io
import numpy as np
import trainModel

# Define model parameters:
epochs = 100
batch = 70

# load data and target
data_path = 'Z:/13_deepPhi/'
files_train = ['conn_matrices.mat', 'network_matrices.mat', 'phismax.mat', 'phismeans.mat']

conn_matrices = network_matrices = phismax = phismeans = []

for file in files_train:
    vars()[file[:-4]] = scipy.io.loadmat(data_path + file)['mat']

phismax = phismax.tolist()[0]
phismeans = phismeans.tolist()[0]

"""1D Dense model: reshape data, split into training and validation data, train model"""
# dim = conn_matrices.shape
# conn_matrices = np.reshape(conn_matrices, (dim[0], dim[1] * dim[2]))
# conn_matrices, val_matrices = conn_matrices[:int(phismax.__len__() * 0.8), :], \
#                               conn_matrices[int(phismax.__len__() * 0.8):phismax.__len__(), :]
# phismax, val_phismax = phismax[:int(phismax.__len__() * 0.8)], phismax[int(phismax.__len__() * 0.8):]
#
# trainModel.train_model_1d(epochs, batch, conn_matrices, val_matrices, phismax, val_phismax)

"""2D CNN model: reshape data, split into training and validation data, train model"""
dim = conn_matrices.shape
conn_matrices = conn_matrices.reshape(dim[0], dim[1], dim[2], 1)
conn_matrices, val_matrices = conn_matrices[:int(phismax.__len__() * 0.8), :, :, :], \
                              conn_matrices[int(phismax.__len__() * 0.8):phismax.__len__(), :, :, :]
phismax, val_phismax = phismax[:int(phismax.__len__() * 0.8)], phismax[int(phismax.__len__() * 0.8):]

trainModel.train_model_cnn(epochs, batch, conn_matrices, val_matrices, phismax, val_phismax)
