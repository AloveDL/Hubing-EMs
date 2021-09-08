# create date: 2021/09/07

from sklearn import svm
from sklearn.externals import joblib


def svm_classifier():
    return svm.LinearSVC()


def train(data, label, svm):
    svm.fit(data, label)


# def precise(test_data):
#     model =
