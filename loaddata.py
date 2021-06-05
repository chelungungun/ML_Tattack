import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold

def load_data(dataset_name, folds, rng):
    X = np.load("multi_label_dataset/" + dataset_name + 'x.npy')
    Y = np.load("multi_label_dataset/" + dataset_name + 'y.npy')
    Y = Y.astype(int)
    Y[Y < 0] = 0

    unlabeled = np.where(Y.sum(-1) < 0)[0]
    print("Features shape = {}".format(X.shape))
    print("Label shape = {}".format(Y.shape))

    cv = []
    kf = KFold(n_splits=folds, shuffle=True, random_state=rng)
    for train, test in kf.split(X):
        cv.append(test.tolist())

    try:
        return (unlabeled, cv, csr_matrix(X), csr_matrix(Y))
    except:
        return (unlabeled, cv, X, Y)

def load_data_correct(dataset_name):
    X = np.load("multi_label_dataset/" + dataset_name + 'x.npy')
    Y = np.load("multi_label_dataset/" + dataset_name + 'y.npy')
    Y = Y.astype(int)
    Y[Y < 0] = 0

    return X, Y


def load_y(dataset_name):
    Y = np.load("multi_label_dataset/" + dataset_name + 'y.npy')
    Y = Y.astype(int)
    Y[Y < 0] = 0

    return Y