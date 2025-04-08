#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 00:02:50 2024

@author: Stacy
"""

import os
import pickle

import optuna

import numpy as np

import sklearn


from lightgbm import early_stopping
from lightgbm import log_evaluation
import optuna.integration.lightgbm as lgb

class ObjectiveGBM(object):
    def __init__(self, Train_Feature, Test_Feature, Train_labels, Test_labels):
        self.Train_Feature = Train_Feature
        self.Test_Feature = Test_Feature
        self.Train_labels = Train_labels
        self.Test_labels = Test_labels

    def __call__(self, trial):
        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        dtrain = lgb.Dataset(self.Train_Feature, label=self.Train_labels)
        dval = lgb.Dataset(self.Test_Feature, label=self.Test_labels)
        gbm = lgb.train(param, dtrain, valid_sets=[dtrain, dval],
                        callbacks=[early_stopping(100), log_evaluation(100)],)
        preds = gbm.predict(self.Test_Feature)
        pred_labels = np.rint(preds)
        # print('pl',pred_labels)
        accuracy = sklearn.metrics.accuracy_score(
            self.Test_labels, pred_labels)
        return accuracy


def go_optLightGBM(Train_Feature, Test_Feature, Train_labels, Test_labels, tarFile, tarpkl):

    if os.path.exists(tarFile):
        best_params = np.load(tarFile, allow_pickle=True)
        best_params = best_params.item()
        # tarpkl = SavePath + "Model_SVM_best_params.pkl"
        if os.path.exists(tarpkl):
            pickfile = open(tarpkl, 'rb')
            gbm = pickle.load(pickfile)
            pickfile.close()
            # preds = gbm.predict(Test_Feature)
            y_probs = gbm.predict_proba(Test_Feature)[:, 1]
            # pred_labels = np.rint(preds)
            accuracy = gbm.score(Test_Feature, Test_labels)
            # accuracy = sklearn.metrics.accuracy_score(Test_labels, pred_labels)

            print('acc:', accuracy)
            return accuracy, best_params, y_probs
        print(best_params)
        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": best_params["lambda_l1"],
            "lambda_l2": best_params["lambda_l2"],
            "num_leaves": best_params["num_leaves"],
            "feature_fraction": best_params["feature_fraction"],
            "bagging_fraction": best_params["bagging_fraction"],
            "bagging_freq": best_params["bagging_freq"],
            "min_child_samples": best_params["min_child_samples"],
        }
        dtrain = lgb.Dataset(Train_Feature, label=Train_labels)
        dval = lgb.Dataset(Test_Feature, label=Test_labels)
        gbm = lgb.train(param, dtrain, valid_sets=[dtrain, dval],
                        callbacks=[early_stopping(100), log_evaluation(100)],)
        # preds = gbm.predict(Test_Feature)
        y_probs = gbm.predict_proba(Test_Feature)[:, 1]
        # pred_labels = np.rint(preds)
        # accuracy = sklearn.metrics.accuracy_score(Test_labels, pred_labels)
        accuracy = gbm.score(Test_Feature, Test_labels)

        print('acc:', accuracy)
        pickfile = open(tarpkl, 'wb')
        pickle.dump(gbm, pickfile)
        pickfile.close()
        return accuracy, best_params, y_probs

    objective = ObjectiveGBM(
        Train_Feature, Test_Feature, Train_labels, Test_labels)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, n_jobs=-1)
    print(study.best_trial)
    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": study.best_params["lambda_l1"],
        "lambda_l2": study.best_params["lambda_l2"],
        "num_leaves": study.best_params["num_leaves"],
        "feature_fraction": study.best_params["feature_fraction"],
        "bagging_fraction": study.best_params["bagging_fraction"],
        "bagging_freq": study.best_params["bagging_freq"],
        "min_child_samples": study.best_params["min_child_samples"],
    }
    dtrain = lgb.Dataset(Train_Feature, label=Train_labels)
    dval = lgb.Dataset(Test_Feature, label=Test_labels)
    gbm = lgb.train(param, dtrain, valid_sets=[dtrain, dval],
                    callbacks=[early_stopping(100), log_evaluation(100)],)
    y_probs = gbm.predict_proba(Test_Feature)[:, 1]
    # preds = gbm.predict(Test_Feature)
    np.save(tarFile, study.best_params)
    return study.best_value, study.best_params, y_probs