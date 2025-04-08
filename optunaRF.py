#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:25:46 2024

@author: Stacy
"""

from sklearn.ensemble import RandomForestClassifier
import optuna
import os
import pickle
import numpy as np

class ObjectiveRF(object):
    def __init__(self, Train_Feature, Test_Feature, Train_labels, Test_labels):
        self.Train_Feature = Train_Feature
        self.Test_Feature = Test_Feature
        self.Train_labels = Train_labels
        self.Test_labels = Test_labels

    def __call__(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        min_samples_split = trial.suggest_float("min_samples_split", 0.1, 1.0)
        min_samples_leaf = trial.suggest_float("min_samples_leaf", 0.1, 0.5)

        classifier_obj = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

        classifier_obj.fit(self.Train_Feature, self.Train_labels)
        accuracy = classifier_obj.score(self.Test_Feature, self.Test_labels)
        return accuracy

def trainRandomForest(best_params, Train_Feature, Test_Feature, Train_labels, Test_labels):
    classifier_obj = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=42
    )

    classifier_obj.fit(Train_Feature, Train_labels)
    
    # y_pred = classifier_obj.predict(Test_Feature)
    y_probs = classifier_obj.predict_proba(Test_Feature)[:, 1]
    y_pred = (y_probs >= 0.5).astype(int)
    accuracy = classifier_obj.score(Test_Feature, Test_labels)
    return accuracy, y_pred, y_probs, classifier_obj

def go_optRandomForest(Train_Feature, Test_Feature, Train_labels, Test_labels, tarFile, tarRFpkl, n_trial, bool_retrain):
    if bool_retrain is True:
        if os.path.exists(tarFile):
            os.remove(tarFile)
        if os.path.exists(tarRFpkl):
            os.remove(tarRFpkl)


    if os.path.exists(tarFile):
        best_params = np.load(tarFile, allow_pickle=True)
        best_params = best_params.item()

        if os.path.exists(tarRFpkl):
            pickfile = open(tarRFpkl, 'rb')
            classifier_obj = pickle.load(pickfile)
            pickfile.close()
            y_probs = classifier_obj.predict_proba(Test_Feature)[:, 1]
            # y_pred = classifier_obj.predict(Test_Feature)
            
            y_pred = (y_probs >= 0.5).astype(int)
            accuracy = classifier_obj.score(Test_Feature, Test_labels)
            print('acc:', accuracy)
            return accuracy, best_params, y_pred, y_probs

        accuracy, y_pred, y_probs, classifier_obj = trainRandomForest(best_params, Train_Feature, Test_Feature, Train_labels, Test_labels)
        print('Train acc:', accuracy)
        pickfile = open(tarRFpkl, 'wb')
        pickle.dump(classifier_obj, pickfile)
        pickfile.close()
        return accuracy, best_params, y_probs, y_probs

    objective = ObjectiveRF(Train_Feature, Test_Feature, Train_labels, Test_labels)

    study = optuna.create_study(direction = "maximize")
    study.optimize(objective, n_trials = n_trial, n_jobs=-1)
    print(study.best_trial)
    accuracy, y_pred, y_probs, classifier_obj = trainRandomForest(study.best_params, Train_Feature, Test_Feature, Train_labels, Test_labels)

    np.save(tarFile, study.best_params)

    return study.best_value, study.best_params, y_pred, y_probs
