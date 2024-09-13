from sklearn.naive_bayes import MultinomialNB
import Levenshtein
from fasta_reader import read_fasta
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, matthews_corrcoef

import numpy as np
import os
import pickle
import sklearn

def column_max_min_normalization(data):
    # Calculate the maximum and minimum values along each column
    max_values = np.max(data, axis=0)
    min_values = np.min(data, axis=0)

    normalized_data = data
    # Perform column-wise max-min normalization
    for i in range(data.shape[1]):
        if (max_values[i] - min_values[i]) != 0:
            normalized_data[:, i] = (
                data[:, i] - min_values[i]) / (max_values[i] - min_values[i])

    return normalized_data
def mypca(fMatrix, n_components):
    n_samples, n_features = fMatrix.shape
    n_components = min(n_samples, n_features) - 1
    pca_model = PCA(n_components = n_components)
    pca_model.fit(fMatrix)
    fMatrix2 = pca_model.fit_transform(fMatrix)
    return fMatrix2

# DataSetDir = data_set[0]
# x_nr_method = nr_method[2] #node representation
# x_classifier = classifier[2]
def get_kfold_index(n, rseed, n_splits=5):
    np.random.seed(rseed)  # 设置随机种子
    indices = np.zeros(n, dtype=int)  # 存储所有折的索引信息
    # 创建KFold对象
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=rseed)
    # 根据KFold对象生成每个折的训练集和测试集索引
    i = 0
    for train_idx, test_idx in kfold.split(np.arange(n)):
        indices[test_idx] = i
        i = i + 1
    return indices


def np2txt(matrixX, output_file):
    # 指定输出文件的路径
    # output_file = "output.txt"

    # 打开文件以写入数据
    with open(output_file, "w") as file:
        # 遍历数组列表
        for array in matrixX:
            # 使用 np.savetxt 将每个数组写入文件
            np.savetxt(file, [array], fmt="%lf", delimiter=",")
            # 添加一个空行以分隔不同的数组（可选）
            # file.write("\n")

    # 关闭文件
    file.close()

    print("Arrays have been written to", output_file)


def plot_roc_curves(tpr_list, fpr_list):
    # def plot_roc_curves(tpr_list, fpr_list, labels,rocs):
    plt.figure(figsize=(8, 8))

    # 自定义线条样式和颜色
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r']
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.plot([1, 0], [0, 1], color='gray', lw=2, linestyle='--')

    for i in range(len(tpr_list)):
        x = fpr_list[i]
        y = tpr_list[i]
        # label = labels[i] if i < len(labels) else f'Fold {i + 1}'
        linestyle = line_styles[i % len(line_styles)]
        color = colors[i % len(colors)]
        # plt.plot(x, y, linestyle=linestyle, color=color, label= label +'=' + str(np.round(rocs[i],4)))
        plt.plot(x, y, linestyle=linestyle, color=color)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(False)

    plt.savefig("output_figure.png", dpi=300, bbox_inches='tight')
    plt.show()

def calculate_metrics_and_roc(y_true, y_pred, y_pred_prob):
    # Convert probability predictions to binary predictions
    # y_true = [0 if x==-1 else x for x in y_true]
    # y_pred = (y_pred_prob >= 0.5).astype(int)

    # Calculate Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Calculate Precision, Recall, and F1-Score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc_score = matthews_corrcoef(y_true, y_pred)
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2,
    #          label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc='lower right')

    # Return computed metrics and ROC curve data
    return acc, precision, recall, f1, mcc_score, roc_auc, fpr, tpr





def compute_Leven(Protein_List):
    Protein_List = Protein_List[:, 1]
    PPSim = np.identity(len(Protein_List))
    for i in range(len(Protein_List)):
        for j in range(i, len(Protein_List)):
            distance = Levenshtein.distance(
                Protein_List[i], Protein_List[j])  # distance指编辑距离
            p = 1 - distance / max(len(Protein_List[i]), len(Protein_List[j]))
            PPSim[i][j] = p
            PPSim[j][i] = p
    return PPSim


def GenFeatureSet(Embedding, PPI_Pos, PPI_Neg, index_pos, index_neg, cv):
    # Embedding = column_max_min_normalization(Embedding)
    
    # Embedding = mypca(Embedding, 64)
    # Embedding = column_max_min_normalization(Embedding)
    train_index_pos = PPI_Pos[np.where(index_pos != cv), :][0]
    train_index_neg = PPI_Neg[np.array(np.where(index_neg != cv)), :][0]
    test_index_pos = PPI_Pos[np.array(np.where(index_pos == cv)), :][0]
    test_index_neg = PPI_Neg[np.array(np.where(index_neg == cv)), :][0]

    TSN_Pos = np.hstack((Embedding[train_index_pos[:, 0], :],
                         Embedding[train_index_pos[:, 1], :]
                         ))

    TSN_Neg = np.hstack((Embedding[train_index_neg[:, 0], :],
                         Embedding[train_index_neg[:, 1], :]
                         ))
    TST_Pos = np.hstack((Embedding[test_index_pos[:, 0], :],
                         Embedding[test_index_pos[:, 1], :]
                         ))
    TST_Neg = np.hstack((Embedding[test_index_neg[:, 0], :],
                         Embedding[test_index_neg[:, 1], :]
                         ))

    Train_Feature = np.vstack((TSN_Pos, TSN_Neg))
    Test_Feature = np.vstack((TST_Pos, TST_Neg))

    Train_labels = np.hstack((np.ones(len(TSN_Pos), dtype=(int)),
                              np.zeros(len(TSN_Neg), dtype=(int))
                              ))
    Test_labels = np.hstack((np.ones(len(TST_Pos), dtype=(int)),
                             np.zeros(len(TST_Neg), dtype=(int))
                             ))
    return Train_Feature, Test_Feature, Train_labels, Test_labels


# def clf_svm(Train_Feature, Test_Feature, Train_labels, Test_labels, best_svc_c):
#     classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
#     classifier_obj.fit(Train_Feature, Train_labels)  # Train the classifier with the chosen hyperparameters
#     predictions = classifier_obj.predict(Test_Feature)
#     accuracy = sklearn.metrics.accuracy_score(Test_labels, predictions)
#     return accuracy


def GetParameterMatrix():
    alp_list = np.linspace(0.5, 1, 5)
    beta_list = np.linspace(0.7, 1, 4)
    # t_list = np.linspace(1, 7, 4).astype(int)
    t_list = [1]

    ParameterMatrix = [[alp, beta, t]
                       for alp in alp_list for beta in beta_list for t in t_list]

    return ParameterMatrix


def GetProteinInfo(firDir):
    Protein_Header = []
    Protein_Sequence = []
    for item in read_fasta(firDir):
        Protein_Header.append(item.defline)
        Protein_Sequence.append(item.sequence)
    return np.stack((Protein_Header, Protein_Sequence), axis=1)


def gaussian_similarity_matrix(data_matrix, gamma=1.0):
    n = data_matrix.shape[0]
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = np.exp(-gamma *
                                             np.linalg.norm(data_matrix[i] - data_matrix[j]) ** 2)
    return similarity_matrix


def gaussian_similarity(matrix):
    num_rows, num_cols = matrix.shape

    Gaussian_similarities = np.zeros((num_rows, num_rows))
    pare_a = 0
    sum_a = 0

    for i in range(num_rows):
        temp = np.linalg.norm(matrix[i, :])
        sum_a += temp ** 2

    pare_a = 1 / (sum_a / num_rows)

    for i in range(num_rows):
        for j in range(num_rows):
            diff = matrix[i, :] - matrix[j, :]
            Gaussian_similarities[i,
            j] = np.exp(-pare_a * np.linalg.norm(diff) ** 2)
    return Gaussian_similarities
    # DS = Gaussian_similarities

    # Gaussian_similarities = np.zeros((num_cols, num_cols))
    # pare_b = 0
    # sum_b = 0

    # for i in range(num_cols):
    #     temp = np.linalg.norm(matrix[:, i])
    #     sum_b += temp**2

    # pare_b = 1 / (sum_b / num_cols)

    # for i in range(num_cols):
    #     for j in range(num_cols):
    #         diff = matrix[:, i] - matrix[:, j]
    #         Gaussian_similarities[i, j] = np.exp(-pare_b * np.linalg.norm(diff)**2)

    # RS = Gaussian_similarities

    # return Gaussian_similarities


def go_NB(Train_Feature, Test_Feature, Train_labels, Test_labels):
    clf = MultinomialNB()
    print(Train_labels)
    print(np.max(Train_Feature),np.min(Train_Feature))
    clf.fit(Train_Feature, Train_labels)
    # y_probs = clf.predict_proba(Test_Feature)
    pred_labels = clf.predict(Test_Feature)
    y_probs = clf.predict_proba(Test_Feature)[:, 1]

    accuracy = sklearn.metrics.accuracy_score(Test_labels, pred_labels)
    print('acc', accuracy)
    return accuracy, [],pred_labels, y_probs
