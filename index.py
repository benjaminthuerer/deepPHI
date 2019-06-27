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

"""check whether PHI value is 0, 0<x<1, 1, or 1<x. Then, perform first-step model to predict these 4 classes.
In a second step, predict PHI values which are > 1 with regressor model"""
# reshape
dim = conn_matrices.shape
conn_matrices = np.reshape(conn_matrices, (dim[0], dim[1] * dim[2]))

# create matrices and targets with the same amount of categoricals (810)
final_matrix = []
min_count = 810  # minimum across all categories
cat_phismax = []
D = {0: 0, 1: 0, 2: 0, 3: 0}
for i, v in enumerate(phismax):
    if v == 0 and D[v] < min_count:
        cat_phismax.append(0)
        final_matrix.append(list(conn_matrices[i, :]))
        D[v] += 1
    elif 0 < v < 1 and D[1] < min_count:
        cat_phismax.append(1)
        final_matrix.append(list(conn_matrices[i, :]))
        D[1] += 1
    elif v == 1 and D[2] < min_count:
        cat_phismax.append(2)
        final_matrix.append(list(conn_matrices[i, :]))
        D[2] += 1
    elif v > 1 and D[3] < min_count:
        cat_phismax.append(3)
        final_matrix.append(list(conn_matrices[i, :]))
        D[3] += 1
    else:
        pass

final_matrix = np.array(final_matrix)

# shuffle data
c = np.c_[final_matrix, cat_phismax]
np.random.shuffle(c)
final_matrix = c[:, :160]
cat_phismax = c[:, 160:]

# learn categoricals and predict PHI values > 1
conn_matrices, val_matrices = final_matrix[:int(cat_phismax.__len__() * 0.8), :], \
                              final_matrix[int(cat_phismax.__len__() * 0.8):cat_phismax.__len__(), :]
phismax, val_phismax = cat_phismax[:int(cat_phismax.__len__() * 0.8)], cat_phismax[int(cat_phismax.__len__() * 0.8):]

trainModel.train_model_categorical(epochs, batch, conn_matrices, val_matrices, phismax, val_phismax)

"""1D Dense model: reshape data, split into training and validation data, train model"""
# dim = conn_matrices.shape
# conn_matrices = np.reshape(conn_matrices, (dim[0], dim[1] * dim[2]))
# conn_matrices, val_matrices = conn_matrices[:int(phismax.__len__() * 0.8), :], \
#                               conn_matrices[int(phismax.__len__() * 0.8):phismax.__len__(), :]
# phismax, val_phismax = phismax[:int(phismax.__len__() * 0.8)], phismax[int(phismax.__len__() * 0.8):]
#
# trainModel.train_model_1d(epochs, batch, conn_matrices, val_matrices, phismax, val_phismax)

"""2D CNN model: reshape data, split into training and validation data, train model"""
# dim = conn_matrices.shape
# conn_matrices = conn_matrices.reshape(dim[0], dim[1], dim[2], 1)
# conn_matrices, val_matrices = conn_matrices[:int(phismax.__len__() * 0.8), :, :, :], \
#                               conn_matrices[int(phismax.__len__() * 0.8):phismax.__len__(), :, :, :]
# phismax, val_phismax = phismax[:int(phismax.__len__() * 0.8)], phismax[int(phismax.__len__() * 0.8):]
#
# trainModel.train_model_cnn(epochs, batch, conn_matrices, val_matrices, phismax, val_phismax)
