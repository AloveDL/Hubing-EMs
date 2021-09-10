# create date: 2021/09/07

from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import KFold  # 主要用于K折交叉验证
from tqdm import tqdm
def svm_classifier(path=None):
    if path is None:
        return svm.SVC(decision_function_shape='ovo',kernel='rbf',C=1)
    else:
        return load_model(path)


def train(data, label, svm_model):
    svm_model.fit(data, label)
    k = 255

    kf = KFold(n_splits=k, random_state=0, shuffle=True)

    # 保存当前最好的k值和对应的准确率
    best_score = 0
    for epoch in range(100):
        curr_score = 0
        for train_index, valid_index in tqdm(kf.split(data)):
            # 每一折的训练以及计算准确率
            # print(train_index)
            # print(train_index,valid_index)
            svm_model.fit([data[i] for i in train_index], [label[i] for i in train_index])
            curr_score = curr_score + svm_model.score([data[i] for i in valid_index], [label[i] for i in valid_index])
            # print(curr_score)
        # 求一下5折的平均准确率
        avg_score = curr_score / k
        if avg_score > best_score:
            # best_k = k
            best_score = avg_score
            save_model(svm_model)
        print("current best score is :%.2f" % best_score)
#
    # print("after cross validation, the final best k is :%d" % best_k)

    print("Model saved Success!")


def predict(test_data, svm_model):
    return svm_model.predict(test_data)


def load_model(path):
    return joblib.load(path)


def save_model(svm_model):
    joblib.dump(svm_model, "model/svm_model.m")
