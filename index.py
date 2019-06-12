import scipy.io

data_path = 'Z:/13_deepPhi/'
files_train = ['conn_matrices.mat', 'network_matrices.mat', 'phismax.mat', 'phismeans.mat']

conn_matrices = network_matrices = phismax = phismeans = []

for file in files_train:
    vars()[file[:-4]] = scipy.io.loadmat(data_path + file)['mat']

phismax = phismax.tolist()[0]
phismeans = phismeans.tolist()[0]
