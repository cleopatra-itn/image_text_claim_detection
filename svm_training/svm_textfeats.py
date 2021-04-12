from sklearn import model_selection, preprocessing, metrics, svm
from sklearn import decomposition
from scipy.special import softmax
import pandas as pd
import numpy as np

from thundersvm import *

import json

import argparse

parser = argparse.ArgumentParser(description='Training SVM')
parser.add_argument('--normalize', type=int, default=1,
                    help='0,1')
parser.add_argument('--tfeat', type=str, default='sumavg',
                    help='sumavg | catavg | last2 | last')
parser.add_argument('--tmodel', type=str, default='bert-base-uncased',
                    help='bert-base-uncased | bertweet | covid_twitter | arabert | bertarabic')
parser.add_argument('--ttype', type=str, default='clean',
                    help='clean | ht | raw | pp (arabic)')
parser.add_argument('--dset', type=str, default='clef_en',
                    help='clef_en | clef_ar | mediaeval | lesa')
parser.add_argument('--split', type=int, default=0,
                    help='0-4')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='0,1,2,3')

args = parser.parse_args()


def get_best_svm_model(feature_vector_train, label_tr, feature_vector_valid, label_vl):
    param_grid = [{'kernel':'rbf', 'C': np.logspace(-1, 1, 15),
                  'gamma': np.logspace(-2, 1, 15)}]

    pca_list = [1.00,0.99,0.98,0.97,0.96,0.95]
    best_acc = 0.0
    best_model = 0
    best_fsc = 0.0
    best_pca_nk = 0
    temp_xtrain = feature_vector_train
    temp_xval = feature_vector_valid
    for pca_nk in pca_list:
        print(pca_nk)
        if pca_nk != 1.0:
            pca = decomposition.PCA(n_components=pca_nk).fit(temp_xtrain)
            feature_vector_train = pca.transform(temp_xtrain)
            feature_vector_valid = pca.transform(temp_xval)
        for params in param_grid:
            for C in params['C']:
                for gamma in params['gamma']:
                    # Model with different parameters
                    model = SVC(C=C, gamma=gamma, kernel=params['kernel'], random_state=42, 
                                    class_weight='balanced', gpu_id=args.gpu_id, max_iter=1000)

                    # fit the training dataset on the classifier
                    model.fit(feature_vector_train, label_tr)

                    preds = model.predict(feature_vector_valid)
                    # predict the acc on validation dataset
                    acc = metrics.accuracy_score(label_vl, preds)
                    fsc = metrics.f1_score(label_vl, preds, average='weighted')

                    if round(acc,4) >= round(best_acc,4):
                        best_acc = acc
                        best_model = model
                        best_pca_nk = pca_nk
                        best_fsc = fsc

    return best_acc, best_fsc, best_pca_nk, best_model



tmodel = args.tmodel
split = args.split
dset = args.dset
tfeat = args.tfeat
ttype = args.ttype

if dset == 'clef_en':
    data_loc = 'data/clef/english/splits/'
elif dset == 'clef_ar':
    data_loc = 'data/clef/arabic/splits/'
elif dset == 'mediaeval':
    data_loc = 'data/mediaeval/splits/'
else:
    data_loc = 'data/lesa/splits/'

tr_df = pd.read_csv(data_loc+'train_%d.txt'%(split), header=None)
vl_df = pd.read_csv(data_loc+'val.txt', header=None)
te_df = pd.read_csv(data_loc+'test_%d.txt'%(split), header=None)

test_idxs = np.array([idx for idx in te_df[0]])

if tmodel in ['bertarabic', 'bertweet', 'covid_twitter']:
    feat_dict  = json.load(open('features/text/%s_%s/%s.json'%(dset, tmodel, tfeat), 'r'))
else:
    feat_dict  = json.load(open('features/text/%s_%s_%s/%s.json'%(dset, tmodel, ttype, tfeat), 'r'))


ft_train = np.array([feat_dict[str(idx)] for idx in tr_df[0]])
ft_val = np.array([feat_dict[str(idx)] for idx in vl_df[0]])
ft_test = np.array([feat_dict[str(idx)] for idx in te_df[0]])

lab_train = tr_df[1].to_numpy()
lab_val = vl_df[1].to_numpy()
lab_test = te_df[1].to_numpy()

if args.normalize:
    ft_train = preprocessing.normalize(ft_train, axis=1)
    ft_val = preprocessing.normalize(ft_val, axis=1)
    ft_test = preprocessing.normalize(ft_test, axis=1)

print(ft_train.shape, ft_val.shape, ft_test.shape)

accuracy, f1_score, best_pca_nk, classifier = get_best_svm_model(ft_train, lab_train, ft_val, lab_val)

if best_pca_nk != 1.0:
    pca = decomposition.PCA(n_components=best_pca_nk).fit(ft_train)
    ft_train = pca.transform(ft_train)
    ft_val = pca.transform(ft_val)
    ft_test = pca.transform(ft_test)

test_preds = classifier.predict(ft_test)
val_preds = classifier.predict(ft_val)
train_preds = classifier.predict(ft_train)

print("SVM %s, Split-%d,  %s-%s-%s"%(dset, split, ttype, tmodel, tfeat))
print("PCA No. Components: %.2f, Dim: %d, SV: %d"%(best_pca_nk, ft_val.shape[1], len(classifier.support_)))
print("C: %.3f, Gamma: %.3f, kernel: %s\n"%(classifier.C, classifier.gamma, classifier.kernel))
print("Train Accuracy: %.4f, Train F1-Score: %.4f"%(round(metrics.accuracy_score(lab_train, train_preds),4),
                        round(metrics.f1_score(lab_train, train_preds, average='weighted'),4)))
print(metrics.confusion_matrix(lab_train, train_preds, labels=[0,1]))
print("Val Accuracy: %.4f, Val F1-Score: %.4f"%(round(accuracy,4), round(f1_score,4)))
print(metrics.confusion_matrix(lab_val, val_preds, labels=[0,1]))
print("Test Accuracy: %.4f, Test F1-Score: %.4f"%(
                        round(metrics.accuracy_score(lab_test, test_preds),4),
                        round(metrics.f1_score(lab_test, test_preds, average='weighted'),4)))
print(metrics.confusion_matrix(lab_test, test_preds, labels=[0,1]))
print('\n')
