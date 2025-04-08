from tools import *
from MyClass import MyClass
from optunaRF import *
from optunaXGB import *
from optunaSVM import *

def GenObj(DataSetDir ,kfCV):
    FileProtein_pos = './Data/' + DataSetDir + '/PPI_No_Pos.txt'
    FileProtein_neg = './Data/' + DataSetDir + '/PPI_No_Neg.txt'
    FileProteinFasta = './Data/' + DataSetDir + '/Sequence.fa'
    FileProteinSim = './Data/' + DataSetDir + '/Levenshtein_Sim.npy'

    # if __name__ == '__main__':

    PPI_Pos = np.loadtxt(FileProtein_pos, dtype=np.int32, delimiter='\t')
    PPI_Neg = np.loadtxt(FileProtein_neg, dtype=np.int32, delimiter='\t')
    PPI_Pos = PPI_Pos - 1
    PPI_Neg = PPI_Neg - 1

    if os.path.exists(FileProteinSim):
        print('Loading Protein Similarity')
        PPSim = np.load(FileProteinSim)
        tarFile = './Data/' + DataSetDir + '/Levenshtein_Sim.csv'
        # np2txt(PPSim, tarFile)
    else:
        print('calulating Protein Similarity')
        PPSim = compute_Leven(GetProteinInfo(FileProteinFasta))
        np.save(FileProteinSim, PPSim)

    ob = MyClass(PPI_Pos, PPI_Neg, PPSim, kfCV=kfCV, DataSetDir=DataSetDir)
    return ob
def go_TrainCLF(x_classifier, Train_Feature, Test_Feature, Train_labels, Test_labels, tarFile, tarpkl, n_trials,
                bool_retrain, n):
    if x_classifier == 'SVM':
        return go_optSVM(
            Train_Feature, Test_Feature, Train_labels, Test_labels, tarFile, tarpkl, n_trials, bool_retrain)
    elif x_classifier == 'XGB':
        return go_optXGB(
            Train_Feature, Test_Feature, Train_labels, Test_labels, tarFile, tarpkl, n_trials, bool_retrain, n=n)
    elif x_classifier == 'NB':
        return go_NB(
            Train_Feature, Test_Feature, Train_labels, Test_labels)
    elif x_classifier == 'RF':
        return go_optRandomForest(
            Train_Feature, Test_Feature, Train_labels, Test_labels, tarFile, tarpkl, n_trials, bool_retrain)

def go_obtainFeature(x_nr_method , ob):
    if x_nr_method == 'fusion_matrix':
        tarQ = ob.SavePath + 'Q_neg.npy'
        if os.path.exists(tarQ):
            Q_neg = np.load(tarQ)
            tarQ = ob.SavePath + 'Q_pos.npy'
            Q_pos = np.load(tarQ)
        else:
            Q_pos, Q_neg = ob.GenFus_Mat()
        x_feature = np.hstack((Q_pos, Q_neg))
    elif x_nr_method == 'FM2DW':
        tarFile = ob.SavePath + 'FM2DW_Embedding_pos_neg.npy'
        if os.path.exists(tarFile):
            x_feature = np.load(tarFile)
        else:
            Q_pos, Q_neg = ob.GenFus_Mat()
            x_feature = ob.GenSTEmbedding(np.hstack((Q_pos, Q_neg)),tarFile)
    return x_feature