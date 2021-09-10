from utils.SVM import svm_classifier, train, predict,load_model
from datasets.dataset import load_label, load_lbp_feature

if __name__ == '__main__':
    svm = svm_classifier()
    x = load_lbp_feature('data/CASME2_lbp_xy_data.txt')
    y = load_label('data/CASME2_label.txt')
    # svm = load_model('model/svm_model.m')
    train(x, y, svm)
