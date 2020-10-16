import pandas as pd
import numpy as np
from config import config as cfg
import os
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split

data_path = './datasets/'

data_file = {
    'heart': 'heart.dat',
    'cleveland': 'cleveland.csv',
    'dermatology': 'dermatology.data',
    'ionosphere': 'ionosphere.data',
    'wine': 'wine.data',
    'vehicle': 'vehicle.csv',
    'sonar': 'sonar.all-data',
    'glass': 'glass.data',
    'segmentation': 'segmentation.test',
    'hepatitis': 'hepatitis.data',
}


def get_CA(vector, X, Y):
    X = X[:, np.where(vector == 1)[0]]

    if X.shape[1] == 0:
        return 0

    if cfg.classifier in ['1NN', '3NN', '5NN']:
        model = KNeighborsClassifier(int(cfg.classifier[0]))
    elif cfg.classifier == 'SVM':
        model = SVC(kernel='rbf')
    elif cfg.classifier == 'C4.5':
        model = DecisionTreeClassifier()  # C4.5 is very similar to CART

    if cfg.fitness_type in ['10-fold', '2-fold']:
        Acc = []
        for train_index, test_index in KFold(n_splits=int(cfg.fitness_type.split('-')[0]), shuffle=True).split(X):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            Acc.append(np.mean(model.fit(X_train, Y_train).predict(X_test) == Y_test))
        return np.array(Acc).mean()
    elif cfg.fitness_type == '7-3':
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)
        return np.mean(model.fit(X_train, Y_train).predict(X_test) == Y_test)


def get_DR(vector):
    return np.mean(vector == 0)


def load_data(data='glass'):
    if data == 'heart':
        data = pd.read_table(os.path.join(data_path, data_file[data]), header=None, sep=' ').values
        X, Y = data[:, :-1], data[:, -1]
    elif data in ['vehicle', 'cleveland']:
        data = pd.read_csv(os.path.join(data_path, data_file[data]), header=None).values
        X, Y = data[:, :-1], data[:, -1]
    elif data == 'dermatology':
        data = pd.read_csv(os.path.join(data_path, data_file[data]), header=None).replace('?', 0).values
        X, Y = data[:, :-1], data[:, -1]
    elif data == 'ionosphere':
        data = pd.read_csv(os.path.join(data_path, data_file[data]), header=None).replace('g', 0).replace('b', 1).values
        X, Y = data[:, :-1], data[:, -1]
    elif data == 'sonar':
        data = pd.read_csv(os.path.join(data_path, data_file[data]), header=None).replace('R', 0).replace('M', 1).values
        X, Y = data[:, :-1], data[:, -1]
    elif data == 'glass':
        data = pd.read_csv(os.path.join(data_path, data_file[data]), header=None).values
        X, Y = data[:, 1:-1], data[:, -1]
    elif data == 'wine':
        data = pd.read_csv(os.path.join(data_path, data_file[data]), header=None).values
        X, Y = data[:, 1:], data[:, 0]
    elif data == 'segmentation':
        data = pd.read_csv(os.path.join(data_path, data_file[data]), header=None)
        for idx, x in enumerate(set(data.iloc[:, 0])):
            data.replace(x, idx, inplace=True)
        data = data.values
        X, Y = data[:, 1:], data[:, 0]
    elif data == 'hepatitis':
        data = pd.read_csv(os.path.join(data_path, data_file[data]), header=None).replace('?', 0).values
        X, Y = data[:, 1:], data[:, 0]
    
    return scale(X), Y.astype('int')


if __name__ == "__main__":
    X, Y = load_data(cfg.data)
    print(X.shape, Y.shape)
