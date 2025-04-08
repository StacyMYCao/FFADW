#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:52:11 2024

@author: Stacy
"""

import numpy as np
import os
import pickle
import sklearn
from sklearn.svm import SVC
import optuna

class Objective(object):
    def __init__(self, Train_Feature, Test_Feature, Train_labels, Test_labels, ):
        self.Train_Feature = Train_Feature
        self.Test_Feature = Test_Feature
        self.Train_labels = Train_labels
        self.Test_labels = Test_labels

    def __call__(self, trial):
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        svc_g = trial.suggest_float("svc_g", 1e-10, 1, log=True)
        degree = trial.suggest_int("degree", 1, 5)
        classifier_obj = SVC(kernel="rbf", C=svc_c, gamma=svc_g, degree=degree,probability=True)
        classifier_obj.fit(self.Train_Feature, self.Train_labels)
        accuracy = classifier_obj.score(self.Test_Feature, self.Test_labels)
        return accuracy

def trainSVM(best_params,Train_Feature, Test_Feature, Train_labels, Test_labels):
    classifier_obj = SVC(kernel="rbf", C=best_params["svc_c"],
                         gamma=best_params["svc_g"],
                         degree=best_params["degree"],
                         probability=True)
    classifier_obj.fit(Train_Feature, Train_labels)
    y_pred=classifier_obj.predict(Test_Feature)
    y_probs = classifier_obj.predict_proba(Test_Feature)[:,1]
    accuracy = classifier_obj.score(Test_Feature, Test_labels)
    return accuracy, y_pred, y_probs, classifier_obj
    

def go_optSVM(Train_Feature, Test_Feature, Train_labels, Test_labels, tarFile, tarSVMpkl, n_trial, bool_retrain):

    if os.path.exists(tarFile):
        best_params = np.load(tarFile, allow_pickle=True)
        best_params = best_params.item()
        
        if os.path.exists(tarSVMpkl):
            pickfile = open(tarSVMpkl, 'rb')
            classifier_obj = pickle.load(pickfile)
            pickfile.close()
            y_probs = classifier_obj.predict_proba(Test_Feature)[:,1]
            y_pred = classifier_obj.predict(Test_Feature)
            accuracy = classifier_obj.score(Test_Feature, Test_labels)
            print(y_probs)
            return accuracy, best_params, y_pred, y_probs
        # print(best_params)
        accuracy, y_pred, y_probs, classifier_obj = trainSVM(best_params,Train_Feature, Test_Feature, Train_labels, Test_labels)
        print('Train acc:', accuracy)
        pickfile = open(tarSVMpkl, 'wb')
        pickle.dump(classifier_obj, pickfile)
        pickfile.close()
        return accuracy, best_params, y_pred, y_probs

    objective = Objective(Train_Feature, Test_Feature, Train_labels, Test_labels)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trial, n_jobs=12)
    print(study.best_trial)
    np.save(tarFile, study.best_params)
    best_params = np.load(tarFile, allow_pickle=True)
    best_params = best_params.item()
    accuracy, y_pred, y_probs, classifier_obj = trainSVM(best_params,Train_Feature, Test_Feature, Train_labels, Test_labels)
    np.save(tarFile, study.best_params)
    print('Train acc:', accuracy)
    pickfile = open(tarSVMpkl, 'wb')
    pickle.dump(classifier_obj, pickfile)
    pickfile.close()
    return study.best_value, study.best_params, y_pred, y_probs