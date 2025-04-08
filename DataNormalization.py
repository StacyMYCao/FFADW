#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:54:35 2024

@author: Stacy
"""


from scipy.stats import entropy
from sklearn.decomposition import PCA

import numpy as np

def mypca(fMatrix, n_components):
    n_samples, n_features = fMatrix.shape
    n_components = min(n_samples, n_features) - 1
    pca_model = PCA(n_components=n_components)
    pca_model.fit(fMatrix)
    fMatrix2 = pca_model.fit_transform(fMatrix)
    return fMatrix2

def column_z_score_normalization(data):
    # Calculate the mean and standard deviation along each column
    mean_values = np.mean(data, axis=0)
    std_values = np.std(data, axis=0)

    # Perform column-wise Z-score normalization
    normalized_data = (data - mean_values) / std_values

    return normalized_data


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

# Variance


def matrix_variance(matrix):
    variances = np.var(matrix, axis=0)  # Compute variance along columns
    normalized_variances = (variances - np.min(variances)) / \
        (np.max(variances) - np.min(variances))
    return normalized_variances

# Example usage:
# variance_scores = matrix_variance(info_matrix)
# print(variance_scores)


#Correlation:
def matrix_correlation(matrix):
    correlation_matrix = np.corrcoef(matrix, rowvar=False)
    normalized_correlation_matrix = (
        correlation_matrix + 1) / 2  # Normalize to range [0, 1]
    return normalized_correlation_matrix

# Example usage:
# correlation_scores = matrix_correlation(info_matrix)
# print(correlation_scores)


#Principal Component Analysis (PCA):
def matrix_pca(matrix, num_components=None):
    pca = PCA(n_components=num_components)
    pca.fit(matrix)
    explained_variance_ratio = pca.explained_variance_ratio_
    normalized_explained_variance_ratio = (explained_variance_ratio - np.min(
        explained_variance_ratio)) / (np.max(explained_variance_ratio) - np.min(explained_variance_ratio))
    return pca.components_, normalized_explained_variance_ratio

# Example usage:
#components, explained_variance_scores = matrix_pca(info_matrix)
#print(components)
#print(explained_variance_scores)


#Information Theory Measures:
def matrix_entropy(matrix):
    entropies = entropy(matrix.T)  # Compute entropy along columns
    normalized_entropies = 1 - (entropies / np.max(entropies))
    return normalized_entropies

# Example usage:
#entropy_scores = matrix_entropy(info_matrix)
#print(entropy_scores)
#For statistical tests such as t-tests, ANOVA, or chi-square tests, the p-value alone can be used as a score. Lower p-values generally indicate greater significance. However, please note that statistical significance may not directly reflect the effectiveness of the matrix itself.

# By normalizing the measures within the desired range, you can obtain scores that range from 0 to 1, indicating the effectiveness of the matrix based on the specific measures used.

def NormX(Mx):
    num_raw, _ = Mx.shape
    for i in range(num_raw):
        sum_raw = np.sum(Mx[i, :])
        if sum_raw == 0:
            continue
        Mx[i, :] = Mx[i, :]/sum_raw
    return Mx


def MaxMinM(Mx):
    MMax = np.max(Mx)
    MMin = np.min(Mx)
    Mx = (Mx - MMin)/(MMax-MMin)
    return Mx
