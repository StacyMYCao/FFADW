#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 00:07:09 2024

@author: Stacy
"""



import sys
import time

from graph import Graph
from tadw import TADW
import networkx as nx

from tools import *
import numpy as np


class MyClass(object):
    def __init__(self, PPI_Pos, PPI_Neg, PPSim, kfCV,DataSetDir):
        self.DataSetDir = DataSetDir

        self.PPI_Pos = PPI_Pos
        self.PPI_Neg = PPI_Neg
        self.kfCV = kfCV
        self.index_pos = get_kfold_index(len(PPI_Pos), rseed=1388, n_splits=kfCV)
        self.index_neg = get_kfold_index(len(PPI_Neg), rseed=1389, n_splits=kfCV)
        self.PPSim = PPSim
        self.boolNP = 0

    def setCV(self, cv):
        self.cv = cv

    def setPara(self, alpha, beta, t):

        self.alpha = np.round(alpha, 3)
        self.beta = np.round(beta, 2)
        self.t = int(t)
        self.SavePath = os.getcwd() + '/' + str(int(self.kfCV)) + '_Fold_CV/' + self.DataSetDir + '/Embedding/k_fold_' + str(
            self.cv + 1) + \
                        '/t_' + str(int(self.t)) + '/alp_' + str(self.alpha) + '_beta_' + str(self.beta) + '/'

        if not os.path.exists(self.SavePath):
            os.makedirs(self.SavePath)

        # print('Current Path:', self.SavePath)

    def ConstructPPINetwork(self, ppis, index_list):
        NumSam = len(self.PPSim)
        TarAdj = np.zeros([NumSam, NumSam])
        TarAdj[ppis[index_list, 0], ppis[index_list, 1]] = 1
        TarAdj[ppis[index_list, 1], ppis[index_list, 0]] = 1
        return TarAdj

    def GenMyGraph(self, ppis_pos, ppis_neg, Embedding):
        Gx = Graph()
        G = nx.DiGraph()
        n_node = len(self.PPSim)
        for i in range(n_node):
            G.add_node(i)

        for i in range(len(ppis_pos)):
            src = ppis_pos[i, 0]
            dst = ppis_pos[i, 1]
            G.add_edge(src, dst)
            G.add_edge(dst, src)
            G[src][dst]['weight'] = 1.0
            G[dst][src]['weight'] = 1.0
        for i in range(len(ppis_neg)):
            src = ppis_neg[i, 0]
            dst = ppis_neg[i, 1]
            G.add_edge(src, dst)
            G.add_edge(dst, src)
            G[src][dst]['weight'] = -1.0
            G[dst][src]['weight'] = -1.0

        Gx.read_g(G)
        for i in range(n_node):
            Gx.G.nodes[i]['label'] = 0
            Gx.G.nodes[i]['feature'] = np.array(Embedding[i, :])
        # print('Graph:' + Gx.G)
        return Gx

    # Refine Feature via combining T & Network
    def GenDLNE(self, ppis_pos, ppis_neg, T):
        g = self.GenMyGraph(ppis_pos, ppis_neg, T)
        model = TADW(graph=g, dim=64, lamb=0.2)
        Embedding = np.zeros([len(self.PPSim), 64])
        for node, vec in model.vectors.items():
            Embedding[node, :] = vec
        return Embedding

    def ConstructMatrixQ(self, PPI_Network):
        # A = gaussian_similarity_matrix(PPI_Network, gamma=1.0)
        A = gaussian_similarity(PPI_Network)
        R = self.PPSim
        num_raw, num_col = self.PPSim.shape
        Pt = np.eye(num_raw)
        Q = np.zeros([num_raw, num_col])
        # Cal：pt1 = alph * A +(1-alph) *R
        # Q = Q + np.power(beta,k) * Pt
        for k in range(int(self.t)):
            print('Processing Fusion Matrix layer t :' + str(k))
            Pt = self.alpha * np.dot(Pt, A) + (1 - self.alpha) * R
            if k == 0:
                Q = Pt
            else:
                Q = Q + np.power(self.beta, k + 1) * (Pt)
        return Q

    def GenFus_Mat(self):
        tarQ = self.SavePath + 'Q_pos.npy'
        if os.path.exists(tarQ):
            Q_pos = np.load(tarQ)
        else:
            train_index_pos = np.array(np.where(self.index_pos != self.cv))
            PPI_Network_pos = self.ConstructPPINetwork(self.PPI_Pos, train_index_pos)
            Q_pos = self.ConstructMatrixQ(PPI_Network_pos)
            np.save(tarQ, Q_pos)
        tarQ = self.SavePath + 'Q_neg.npy'

        if os.path.exists(tarQ):
            Q_neg = np.load(tarQ)
        else:
            train_index_neg = np.array(np.where(self.index_neg != self.cv))
            PPI_Network_neg = self.ConstructPPINetwork(self.PPI_Neg, train_index_neg)
            Q_neg = self.ConstructMatrixQ(PPI_Network_neg)
            np.save(tarQ, Q_neg)
        # fus_Mat = np.hstack((Q_pos, Q_neg))
        return Q_pos, Q_neg

    def GenSAEEmbedding(self,Q_pos, Q_neg):
        from SDAE import stacked_denoising_autoencoder_node_embedding
        tarFile = self.SavePath + 'SAE_Embedding_pos_neg.npy'
        if os.path.exists(tarFile):
            Embedding_pos_neg = np.load(tarFile)
        else:
            stime = time.time()
            Embedding_pos_neg, loss_value = stacked_denoising_autoencoder_node_embedding(
            np.hstack((Q_pos, Q_neg)), output_dim=128, encoder_params=[1024, 512], learning_rate=0.001, batch_size=32, epochs=200,
            noise_factor=0.1)

            np.save(tarFile, Embedding_pos_neg)
            print('calculating SAE costing time:', time.time() - stime)
        return Embedding_pos_neg

    def GenSTEmbedding(self, Embedding_pos_neg,tarFile):
        if os.path.exists(tarFile):
            return np.load(tarFile)
        else:
            ST_Embedding = self.GenDLNE(self.PPI_Pos, self.PPI_Neg, Embedding_pos_neg)
            np.save(tarFile, ST_Embedding)
        return ST_Embedding


