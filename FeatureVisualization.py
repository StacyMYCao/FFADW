#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:49:39 2024

@author: Stacy
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity

def show_TSNE_PN(data_pos, data_neg):
    # Concatenate the positive and negative data
    data = np.vstack((data_pos, data_neg))

    tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
    embedded_data = tsne.fit_transform(data)

    # Estimate density using KernelDensity
    # You can adjust the bandwidth as needed
    kde = KernelDensity(bandwidth=0.01)
    kde.fit(embedded_data)
    densities = np.exp(kde.score_samples(embedded_data))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Use colormap to represent densities
    sc_pos = ax.scatter(embedded_data[:len(data_pos), 0], embedded_data[:len(data_pos), 1],
                        embedded_data[:len(data_pos), 2], c=densities[:len(data_pos)], cmap='viridis', marker='o', s=25, label='Positive')
    sc_neg = ax.scatter(embedded_data[len(data_pos):, 0], embedded_data[len(data_pos):, 1],
                        embedded_data[len(data_pos):, 2], c=densities[len(data_pos):], cmap='viridis', marker='x', s=25, label='Negative')
    
    cbar = fig.colorbar(sc_pos)
    cbar.set_label('Density')

    ax.set_title("t-SNE 3D Visualization with Density")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.legend(loc='best')
    plt.show()
    print("Positive Data - Mean Density:", np.mean(densities[:len(data_pos)]))
    print("Positive Data - Std Density:", np.std(densities[:len(data_pos)]))
    print("Negative Data - Mean Density:", np.mean(densities[len(data_pos):]))
    print("Negative Data - Std Density:", np.std(densities[len(data_pos):]))



def show_TSNE_3D(data):
    # tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
    # embedded_data = tsne.fit_transform(data)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2], marker='o', s=25)
    # ax.set_title("t-SNE 3D Visualization")
    # ax.set_xlabel("Dimension 1")
    # ax.set_ylabel("Dimension 2")
    # ax.set_zlabel("Dimension 3")
    # plt.show()
    tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
    embedded_data = tsne.fit_transform(data)

    # Estimate density using KernelDensity
    # You can adjust the bandwidth as needed
    kde = KernelDensity(bandwidth=0.1)
    kde.fit(embedded_data)
    densities = np.exp(kde.score_samples(embedded_data))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Use colormap to represent densities
    sc = ax.scatter(embedded_data[:, 0], embedded_data[:, 1],
                    embedded_data[:, 2], c=densities, cmap='viridis', marker='o', s=25)
    cbar = fig.colorbar(sc)
    cbar.set_label('Density')

    ax.set_title("t-SNE 3D Visualization with Density")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    plt.show()
    print(np.mean(densities))
    print(np.std(densities))


def show_TSNE_2D(X):
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    X_embedded = tsne.fit_transform(X)

    # Compute kernel density estimation to estimate point densities in high-dimensional space
    kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
    kde.fit(X)

    # Estimate densities for each data point
    densities = np.exp(kde.score_samples(X))

    # Plot the t-SNE embeddings colored by densities
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                c=densities, cmap='viridis')
    plt.colorbar()
    plt.title('t-SNE with Point Densities')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()
    print(densities)