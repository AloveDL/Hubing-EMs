# create date: 2021/09/07

from sklearn import svm
from sklearn.externals import joblib


def svm_classifier(path=None):
    if path is None:
        return svm.LinearSVC(loss="squared_hinge", C=1)
    else:
        return load_model(path)


def train(data, label, svm_model):
    svm_model.fit(data, label)
    save_model(svm_model)
    print("Model saved Success!")


def predict(test_data, svm_model):
    return svm_model.predict(test_data)


def load_model(path):
    return joblib.load(path)


def save_model(svm_model):
    joblib.dump(svm_model, "model/svm_model.m")
