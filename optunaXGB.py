#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 00:00:12 2024

@author: Stacy
"""

import os
import pickle

import optuna
import xgboost as xgb

import numpy as np
np.random.seed(2025)
import sklearn
from sklearn.metrics import accuracy_score
class ObjectiveXGB(object):
    def __init__(self, Train_Feature, Test_Feature, Train_labels, Test_labels):
        self.Train_Feature = Train_Feature
        self.Test_Feature = Test_Feature
        self.Train_labels = Train_labels
        self.Test_labels = Test_labels

    def __call__(self, trial):
        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            # use exact for small dataset.
            "tree_method": "exact",
            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            # L2 regularization weight.
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }

        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = trial.suggest_int(
                "min_child_weight", 2, 10)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            # defines how selective algorithm is.
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"])

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical(
                "sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical(
                "normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float(
                "rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float(
                "skip_drop", 1e-8, 1.0, log=True)

        dtrain = xgb.DMatrix(self.Train_Feature, label=self.Train_labels)
        dvalid = xgb.DMatrix(self.Test_Feature, label=self.Test_labels)

        bst = xgb.train(param, dtrain)
        preds = bst.predict(dvalid)        
        # print('preds:',preds)
        pred_labels = (preds>=0.5).astype(int)
        if( np.isnan(preds).any()):
            accuracy = -1
        else:
            accuracy = accuracy_score(self.Test_labels, pred_labels)                
        return accuracy

def GetBsetParam(best_params):
    param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            # use exact for small dataset.
            "tree_method": "exact",
            # defines booster, gblinear for linear functions.
            "booster": best_params["booster"],
            # L2 regularization weight.
            "lambda": best_params["lambda"],
            # L1 regularization weight.
            "alpha": best_params["alpha"],
            # sampling ratio for training data.
            "subsample": best_params["subsample"],
            # sampling according to each tree.
            "colsample_bytree": best_params["colsample_bytree"],
        }
    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = best_params["max_depth"]
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = best_params["min_child_weight"]
        param["eta"] = best_params["eta"]
        # defines how selective algorithm is.
        param["gamma"] = best_params["gamma"]
        param["grow_policy"] = best_params["grow_policy"]

    if param["booster"] == "dart":
        param["sample_type"] = best_params["sample_type"]
        param["normalize_type"] = best_params["normalize_type"]
        param["rate_drop"] = best_params["rate_drop"]
        param["skip_drop"] = best_params["skip_drop"]
    return param

def go_optXGB(Train_Feature, Test_Feature, Train_labels, Test_labels, tarFile, tarpkl, n_trial, bool_retrain, n):
    dtrain = xgb.DMatrix(Train_Feature, label=Train_labels)
    dvalid = xgb.DMatrix(Test_Feature, label=Test_labels)
    if n != 0 or bool_retrain is True:
        # return
        if os.path.exists(tarFile):
            os.remove(tarFile)
        if os.path.exists(tarpkl):
            os.remove(tarpkl)
    elif n==11:
        return 0,0,0,0


    if os.path.exists(tarFile):
        best_params = np.load(tarFile, allow_pickle=True)
        best_params = best_params.item()

        if os.path.exists(tarpkl):
            
            pickfile = open(tarpkl, 'rb')
            bst = pickle.load(pickfile)
            pickfile.close()
            preds = bst.predict(dvalid)
            pred_labels = (preds>=0.5).astype(int)
            accuracy = accuracy_score(Test_labels, pred_labels)
            if accuracy >0.5:
                return accuracy, best_params, pred_labels,preds
            else:
                # param = GetBsetParam(best_params)
                # bst = xgb.train(param, dtrain)
                # preds = bst.predict(dvalid)
                # # pred_labels = np.rint(preds)
                # if( np.isnan(preds).any()):
                #     accuracy = -1
                # else:              
                #     pred_labels = (preds>=0.5).astype(int)
                #     accuracy = accuracy_score(Test_labels, pred_labels)
                #     print('acc:', accuracy)
                #     pickfile = open(tarpkl, 'wb')
                #     pickle.dump(bst, pickfile)
                #     pickfile.close()
                # if accuracy <= 0.97:
                    # if os.path.exists(tarFile):
                    #     os.remove(tarFile)
                    # if os.path.exists(tarpkl):
                        # os.remove(tarpkl)
                accuracy, best_params, pred_labels,preds = go_optXGB(Train_Feature, Test_Feature, Train_labels, Test_labels, tarFile, tarpkl, n_trial, bool_retrain, n+1)
                    
                print(best_params)
                
            return accuracy, best_params, pred_labels, preds
    print('Retraining:', tarFile)
    objective = ObjectiveXGB(
        Train_Feature, Test_Feature, Train_labels, Test_labels)

    study = optuna.create_study(direction="maximize")
    # importance
    study.optimize(objective, n_trials=n_trial, n_jobs=8)
    print(study.best_trial)
    np.save(tarFile, study.best_params)
    best_params = study.best_params
    param = GetBsetParam(best_params)
    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    if( np.isnan(preds).any()):
        accuracy = -1
    else:
        pred_labels = (preds>=0.5).astype(int)
        accuracy = accuracy_score(Test_labels, pred_labels)

    pickfile = open(tarpkl, 'wb')
    pickle.dump(bst, pickfile)
    pickfile.close()
    pickfile = open(tarpkl, 'rb')
    bst = pickle.load(pickfile)
    pickfile.close()
    preds = bst.predict(dvalid)
    if( np.isnan(preds).any()):
        accuracy = -1
    else:
        pred_labels = (preds>=0.5).astype(int)
        accuracy2 = accuracy_score(Test_labels, pred_labels)
        if(accuracy2 == accuracy):
            print('coordinate')
    
    return study.best_value, study.best_params, pred_labels, preds
