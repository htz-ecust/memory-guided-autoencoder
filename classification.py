#%%
import sys
import seaborn as sn
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader,Dataset
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from read_data import create_data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

#%%

#%%
def pre_model(classifier, traindata, train_attributelabel, testdata, label, att):
    clf_dict = {'SVC':SVC(kernel='linear'),'RF':RandomForestClassifier(n_estimators=100),'NB':GaussianNB()}
    res_list = []
    for i in range(train_attributelabel.shape[1]):
        clf = clf_dict[classifier]
        if max(train_attributelabel[:, i]) != 0:
            clf.fit(traindata, train_attributelabel[:, i])
            res = clf.predict(testdata)
        else:
            res = np.zeros(testdata.shape[0])
        res_list.append(res.T)
    test_pre_attribute = np.row_stack(res_list).T

    label_lis = []
    for i in range(test_pre_attribute.shape[0]):
        pre_res = test_pre_attribute[i, :]
        loc = (np.sum(np.square(att - pre_res), axis=1)).argmin()
        label_lis.append(label[loc])
    label_lis = np.row_stack(label_lis).squeeze()
    # label_lis = np.mat(np.row_stack(label_lis))
    # print(model)
    # print(accuracy_score(label_lis, testlabel))
    return test_pre_attribute,label_lis

def our_pre_model(classifier, traindata, train_attributelabel, testdata, testmarks,label,label_,att,att_):
    clf_dict = {'SVC':SVC(kernel='linear'),'RF':RandomForestClassifier(n_estimators=100),'NB':GaussianNB()}
    res_list = []
    for i in range(train_attributelabel.shape[1]):
        clf = clf_dict[classifier]
        if max(train_attributelabel[:, i]) != 0:
            clf.fit(traindata, train_attributelabel[:, i])
            res = clf.predict(testdata)
        else:
            res = np.zeros(testdata.shape[0])
        res_list.append(res.T)
    test_pre_attribute = np.row_stack(res_list).T

    label_lis = []
    for i in range(test_pre_attribute.shape[0]):
        pre_res = test_pre_attribute[i, :]
        if testmarks[i]==1:
            loc = (np.sum(np.square(att - pre_res), axis=1)).argmin()
            label_lis.append(label[loc])
        else:
            loc = (np.sum(np.square(att_ - pre_res), axis=1)).argmin()
            label_lis.append(label_[loc])
    label_lis = np.row_stack(label_lis).squeeze()
    # print(model)
    # print(accuracy_score(label_lis, testlabel))
    return test_pre_attribute,label_lis
#%%
def our_eval(cfg,model,traindatas,train_x,trainatts,trainlabels,valdatas,val_x,val_marks,vallabels,testdatas,test_x,test_marks,testlabels,att_s,att_u,atts):
    
    clf_dict = {'SVC':SVC(kernel='linear'),'RF':RandomForestClassifier(n_estimators=100),'NB':GaussianNB()}
    classifier= cfg.EVALUATION.classifier
    # mode = cfg.EVALUATION.mode
    concate = cfg.EVALUATION.concate
    seen_clf = cfg.EVALUATION.seen_clf
    clf = clf_dict[classifier]

    model.eval()
    with torch.no_grad():
        train_z = model.encoder(traindatas).detach().cpu().numpy()
        val_z = model.encoder(valdatas).detach().cpu().numpy()
        test_z = model.encoder(testdatas).detach().cpu().numpy()

    if concate=='res':
        train_z_ = np.c_[train_x,train_z]
        val_z_ =np.c_[val_x,val_z]
        test_z_ = np.c_[test_x,test_z]
    elif concate=='z':        
        train_z_ = train_z
        val_z_ = val_z
        test_z_ = test_z
    else:
        train_z_ = train_x
        val_z_ = val_x
        test_z_ = test_x


    ## CZSL eveluation
    seen_classes = np.unique(vallabels)
    unseen_classes = np.unique(testlabels)
    testpre_att, testlabel_lis =  pre_model(classifier, train_z_, trainatts, test_z_, unseen_classes,att_u)
    acc = accuracy_score(testlabel_lis, testlabels)
    
    ## GZSL evaluation
    testpre_att,testlabel_lis = our_pre_model(classifier, train_z_, trainatts, test_z_, test_marks,unseen_classes,seen_classes,att_u,att_s)
    unseen_accs = 0
    for i in unseen_classes:
        unseen_accs += accuracy_score(testlabel_lis[testlabels==i],testlabels[testlabels==i])
    U = unseen_accs/unseen_classes.shape[0]
    
    if seen_clf=='supervised':
        clf.fit(train_z_,trainlabels)
        vallabel_lis = clf.predict(val_z_)
        vallabel_lis[val_marks==0] = np.random.choice(unseen_classes)
        seen_accs=0
        for i in seen_classes:
            seen_accs += accuracy_score(vallabel_lis[vallabels==i],vallabels[vallabels==i])
        S = seen_accs/seen_classes.shape[0]
    else:
        valpre_att,vallabel_lis = our_pre_model(classifier, train_z_, trainatts, val_z_, val_marks,seen_classes,unseen_classes,att_s,att_u)
        seen_accs=0
        for i in seen_classes:
            seen_accs += accuracy_score(vallabel_lis[vallabels==i],vallabels[vallabels==i])
        S = seen_accs/seen_classes.shape[0]
    H = 2*(S*U)/(S+U)
    print('Acc: {:.4f}, S: {:.4f}, U: {:.4f}, H: {:.4f}'.format(acc,S,U,H))
    y_pre_label = np.r_[vallabel_lis,testlabel_lis] 
    return acc,U,S,H, train_z,val_z,test_z,y_pre_label


# %%
def FDAT_clf(classifier, traindata, train_attributelabel, testdata, label, att):
    clf_dict = {'SVC':SVC(kernel='linear'),'RF':RandomForestClassifier(),'NB':GaussianNB()}
    res_list = []
    for i in range(train_attributelabel.shape[1]):
        clf = clf_dict[classifier]
        if max(train_attributelabel[:, i]) != 0:
            clf.fit(traindata, train_attributelabel[:, i])
            res = clf.predict(testdata)
        else:
            res = np.zeros(testdata.shape[0])
        res_list.append(res.T)
    test_pre_attribute = np.row_stack(res_list).T

    label_lis = []
    for i in range(test_pre_attribute.shape[0]):
        pre_res = test_pre_attribute[i, :]
        loc = (np.sum(np.square(att - pre_res), axis=1)).argmin()
        label_lis.append(np.arange(15)[loc])
    label_lis = np.row_stack(label_lis).squeeze()
    # print(model)
    # print(accuracy_score(label_lis, testlabel))
    return test_pre_attribute,label_lis

#%%
