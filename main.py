import sys
import time

import numpy as np
from tools import *
from tools_proc import *
from DataNormalization import *
np.random.seed(2025)

kfCV = 5
bool_retrain = False
bool_para = False
n_trials = 64
bool_train = 1
data_set = ['Yeast', 'Human', 'Matine']
#, 'Drosophila']
nr_method = [ 'fusion_matrix','FM2DW']
classifier = [ 'SVM','XGB','RF','NB' ]

if len(sys.argv) < 4:
    print("请至少提供一个参数。")
    sys.exit(1)

DataSetDir = data_set[data_set.index(sys.argv[1])]
x_nr_method = nr_method[nr_method.index(sys.argv[2])] #node representation
x_classifier = classifier[classifier.index(sys.argv[3])]


ob = GenObj(DataSetDir, kfCV = kfCV)

alp_list = np.linspace(0, 1, 9) 
# beta_list = np.linspace(0.7, 1, 4)
# beta_list = np.linspace(1, 10, 9)
# t_list = np.linspace(1, 7, 4).astype(int)

if len(sys.argv) == 5:
    
    alp_list = [float(sys.argv[4])]
    # print("alpha=",sys.argv[4],type(float(alp_list[0])))
    # bool_retrain = True
    bool_retrain = False
    bool_para = False
    n_trials = 512
beta_list = [1]
# alp_list = np.delete(alp_list,np.where(alp_list==0.625))
t_list = [1]
accs = np.zeros([kfCV, len(alp_list)])

scores = np.zeros([kfCV, 6])
pred_list = []
test_list = []
fpr_list = []
tpr_list = []


for cv in range(kfCV):  # kfCV
# for cv in [2]:  # kfCV
    ts=time.time()
    ob.setCV(cv)
    i = 0
    for alpha in (alp_list):
        for beta in (beta_list):
            for t in (t_list):
                #ts = time.time()
                ob.setPara(alpha, beta, t)
            
                tarFile = ob.SavePath + x_nr_method + '_' + x_classifier + '_best_params.npy'
                tarpkl = ob.SavePath + x_nr_method + '_' + x_classifier + '_best_params.pkl'
                

                if bool_retrain is True:
                    if os.path.exists(tarFile):
                        os.remove(tarFile)
                    if os.path.exists(tarpkl):
                        os.remove(tarpkl)

                print('tarFile:', tarFile)
                # if bool_para is True:
                #     os.remove(tarpkl+'training')



                if bool_para is True:
                    if os.path.exists(tarFile) or os.path.exists(tarpkl) or os.path.exists(tarpkl+'training') :
                        # os.remove(tarpkl+'training')
                        print('tarFile:', tarFile ,'training')
                        continue
                    else:
                        file = open(tarpkl+'training', 'w')
                        file.close()
                
                

                if bool_train == 1:
                    x_feature = go_obtainFeature(x_nr_method, ob)
                    
                    Train_Feature, Test_Feature, Train_labels, Test_labels = \
                        GenFeatureSet(x_feature, ob.PPI_Pos, ob.PPI_Neg, \
                                    ob.index_pos, ob.index_neg, ob.cv)
                    # del x_feature

                    accs[cv][i], _, y_pred, y_probs =  go_TrainCLF(x_classifier, Train_Feature, Test_Feature, Train_labels, Test_labels, tarFile, tarpkl, n_trials, bool_retrain, n=0)

                    if len(alp_list) == 1:
                        y_pred = (y_probs >= 0.5).astype(int)
                        scores[cv, 0], scores[cv, 1], scores[cv, 2], scores[cv, 3], scores[cv, 4], scores[cv, 5], fpr, tpr \
                            = calculate_metrics_and_roc(Test_labels, y_pred, y_probs)
                        scores[cv, 0] = accs[cv][i]
                        pred_list.append(y_pred)
                        test_list.append(Test_labels)
                        fpr_list.append(fpr)
                        tpr_list.append(tpr)
                if bool_para is True:
                    os.remove(tarpkl+'training')
    
        i = i+1
    print("time costing", time.time() - ts)

for i in range(accs.shape[0]):
    for j in range(accs.shape[1]):
        print(accs[i][j], end= '\t')
    print()

for i in range(scores.shape[0]):
    for j in range(scores.shape[1]):
        print(scores[i][j], end= '\t')
    print()
if len(alp_list) == 1:
    tartpr = './rec/' +  DataSetDir + '_'+ x_nr_method + '_' + x_classifier + '_' + str(alp_list) + '_tpr_list.csv'
    tarfpr = './rec/' +  DataSetDir + '_'+ x_nr_method + '_' + x_classifier + '_' + str(alp_list) + '_fpr_list.csv'
    np2txt(tpr_list, tartpr)
    np2txt(fpr_list, tarfpr)
print(np.round( np.mean(scores[:, 0]*100),2 ), "±", np.round(np.std(scores[:, 0]*100),2), end= '\t')
print(np.round(np.mean(scores[:, 1]*100),2), "±", np.round(np.std(scores[:, 1]*100),2), end= '\t')
print(np.round(np.mean(scores[:, 2]*100),2), "±", np.round(np.std(scores[:, 2]*100),2), end= '\t')
print(np.round(np.mean(scores[:, 3]*100), 2),"±", np.round(np.std(scores[:, 3]*100),2), end= '\t')
print(np.round(np.mean(scores[:, 4]),4), "±",np.round( np.std(scores[:, 4]),4), end= '\t')
print(np.round(np.mean(scores[:, 5]),4), "±", np.round(np.std(scores[:, 5]),4))