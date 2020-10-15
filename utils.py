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
    'glass': 'glass.data',
    'cleveland': 'cleveland.data'
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
        pred = model.fit(X_train, Y_train).predict(X_test)
        return np.mean(model.fit(X_train, Y_train).predict(X_test) == Y_test)


def get_DR(vector):
    return np.mean(vector == 0)


def load_data(data='glass'):
    if data == 'glass':
        df = pd.read_csv(os.path.join(data_path, data_file[data]), header=None)
        # print(df.head())
        data = df.values
        X, Y = data[:, 1:-1], data[:, -1]
    elif data == 'cleveland':
        df = pd.read_table(os.path.join(data_path, data_file[data]))
    return scale(X), Y.astype('int')


if __name__ == "__main__":
    X, Y = load_data(cfg.data)
    print(X.shape, Y.shape)
