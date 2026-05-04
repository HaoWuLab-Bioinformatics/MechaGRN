import math

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class scRNADataset(Dataset):
    """
    PyTorch Dataset for single-cell RNA-seq GRN training data.

    Args:
        train_set: Training samples with (TF_idx, Target_idx, Label)
        num_gene: Total number of genes in the network
        flag: Whether to use soft labels for classification
    """
    def __init__(self, train_set, num_gene, flag=False):
        super(scRNADataset, self).__init__()
        self.train_set = train_set
        self.num_gene = num_gene
        self.flag = flag

    def __getitem__(self, idx):
        """Get a single training sample."""
        train_data = self.train_set[:, :2]  # TF and Target indices
        train_label = self.train_set[:, -1]  # Regulatory link label

        if self.flag:
            # Convert to soft labels for certain loss functions
            train_len = len(train_label)
            train_tan = np.zeros([train_len, 2])
            train_tan[:, 0] = 1 - train_label
            train_tan[:, 1] = train_label
            train_label = train_tan

        data = train_data[idx].astype(np.int64)
        label = train_label[idx].astype(np.float32)

        return data, label

    def __len__(self):
        """Return total number of samples."""
        return len(self.train_set)

    def Adj_Generate(self, TF_set, direction=False, loop=False):
        """
        Generate adjacency matrix from training regulatory links.

        Args:
            TF_set: Set of transcription factor indices
            direction: Whether to use directed adjacency (default: False)
            loop: Whether to add self-loops (default: False)

        Returns:
            Sparse adjacency matrix in DOK format
        """

        adj = sp.dok_matrix((self.num_gene, self.num_gene), dtype=np.float32)

        for pos in self.train_set:
            tf = pos[0]
            target = pos[1]

            if direction == False:
                # Undirected: both TF→Target and Target→TF
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    adj[target, tf] = 1.0
            else:
                # Directed: only TF→Target, but bidirectional if target is also a TF
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    if target in TF_set:
                        adj[target, tf] = 1.0

        # Add self-loops if specified
        if loop:
            adj = adj + sp.identity(self.num_gene)

        adj = adj.todok()

        return adj


class load_data():
    """
    Data loader for gene expression matrix with normalization.

    Args:
        data: Gene expression DataFrame (genes × cells)
        normalize: Whether to apply standard normalization (default: True)
    """
    def __init__(self, data, normalize=True):
        self.data = data
        self.normalize = normalize

    def data_normalize(self, data):
        """
        Apply standard scaler normalization to expression data.
        Normalizes across cells (columns) for each gene.
        """
        standard = StandardScaler()
        epr = standard.fit_transform(data.T)  # Normalize per gene
        return epr.T

    def exp_data(self):
        """
        Process and return expression feature matrix.

        Returns:
            Normalized expression matrix as float32 numpy array
        """
        data_feature = self.data.values

        if self.normalize:
            data_feature = self.data_normalize(data_feature)

        data_feature = data_feature.astype(np.float32)

        return data_feature


def adj2saprse_tensor(adj):
    """
    Convert scipy sparse matrix to PyTorch sparse tensor.

    Args:
        adj: Scipy sparse adjacency matrix

    Returns:
        PyTorch sparse COO tensor
    """
    coo = adj.tocoo()
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.from_numpy(coo.data).float()

    adj_sp_tensor = torch.sparse_coo_tensor(i, v, coo.shape)

    return adj_sp_tensor


def normalize(expression):
    """
    Apply standard scaler normalization.

    Args:
        expression: Expression matrix to normalize

    Returns:
        Normalized expression matrix
    """
    std = StandardScaler()
    epr = std.fit_transform(expression)

    return epr


def Network_Statistic(data_type, net_scale, net_type):
    """
    Get network density statistics for different benchmark datasets.

    Args:
        data_type: Cell type (e.g., hESC, mESC, mHSC-GM)
        net_scale: Number of genes (500 or 1000)
        net_type: Network type (Non-Specific, Specific, STRING, Lofgof)

    Returns:
        Network density value for the specified dataset
    """

    if net_type == 'Non-Specific':
        # Non-specific network density dictionary
        dic = {'hESC500': 0.016, 'hESC1000': 0.014, 'hHEP500': 0.015, 'hHEP1000': 0.013, 'mDC500': 0.019,
               'mDC1000': 0.016, 'mESC500': 0.015, 'mESC1000': 0.013, 'mHSC-E500': 0.022, 'mHSC-E1000': 0.020,
               'mHSC-GM500': 0.030, 'mHSC-GM1000': 0.029, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.043}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Specific':
        # Specific network density dictionary (higher density)
        dic = {'hESC500': 0.164, 'hESC1000': 0.165, 'hHEP500': 0.379, 'hHEP1000': 0.377, 'mDC500': 0.085,
               'mDC1000': 0.082, 'mESC500': 0.345, 'mESC1000': 0.347, 'mHSC-E500': 0.578, 'mHSC-E1000': 0.566,
               'mHSC-GM500': 0.543, 'mHSC-GM1000': 0.565, 'mHSC-L500': 0.525, 'mHSC-L1000': 0.507}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'STRING':
        # STRING protein-protein interaction network density
        dic = {'hESC500': 0.024, 'hESC1000': 0.021, 'hHEP500': 0.028, 'hHEP1000': 0.024, 'mDC500': 0.038,
               'mDC1000': 0.032, 'mESC500': 0.024, 'mESC1000': 0.021, 'mHSC-E500': 0.029, 'mHSC-E1000': 0.027,
               'mHSC-GM500': 0.040, 'mHSC-GM1000': 0.037, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.045}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Lofgof':
        # Knockout perturbation network density
        dic = {'mESC500': 0.158, 'mESC1000': 0.154}

        query = 'mESC' + str(net_scale)
        scale = dic[query]
        return scale

    else:
        raise ValueError(f"Unknown network type: {net_type}")


def random_walk_imp(matrix, rp):
    """
    Random walk with restart for network propagation.

    Computes the steady-state probability distribution of a random walker
    that restarts from the initial node with probability (1-rp).

    Args:
        matrix: Adjacency matrix for propagation
        rp: Restart probability (default: 0.7)

    Returns:
        Steady-state probability matrix Q
    """
    row, col = matrix.shape
    row_sum = np.sum(matrix, axis=1)
    # Handle zero-degree nodes
    for i in range(row_sum.shape[0]):
        if row_sum[i] == 0:
            row_sum[i] = 0.001
    # Normalize adjacency matrix
    nor_matrix = np.divide(matrix.T, row_sum).T
    Q = np.eye(row)
    I = np.eye(row)
    # Iterate until convergence
    for i in range(30):
        # Random walk iteration: Q_new = rp * Q * P + (1-rp) * I
        Q_new = rp * np.dot(Q, nor_matrix) + (1 - rp) * I
        delta = np.linalg.norm(Q - Q_new)
        Q = Q_new.copy()
        # Convergence threshold
        if delta < 1e-6:
            break
    return Q


def Feature_discretization_data(data, class_num):
    """
    Discretize gene expression data into discrete levels.

    Log-transforms non-zero expression values and assigns them to
    discrete bins based on their distance from the mean.

    Args:
        data: Gene expression DataFrame (genes × cells)
        class_num: Number of discrete levels

    Returns:
        Discretized expression tensor
    """
    data = data.T
    gene_name = list(data)  # Get gene names (column indices)
    new_data_dict = {}

    # Identify zero values for masking
    zero_index = np.where(data.values == 0)
    mask = np.ones_like(data.values)
    mask[zero_index] = 0

    for gene in gene_name:
        temp = data[gene]
        non_zero_element = np.log(temp[temp != 0.].values)

        if len(non_zero_element) == 0:
            # All values are zero for this gene
            new_data_dict[gene] = temp.apply(lambda x: 0)
            continue

        # Calculate discretization boundaries
        mean = np.mean(non_zero_element)
        tmin = np.min(non_zero_element)
        std = np.std(non_zero_element)
        tmax = np.max(non_zero_element)

        # Define bin boundaries within 2 standard deviations
        lower_bound = max(mean - 2 * std, tmin)
        upper_bound = min(mean + 2 * std, tmax)
        bucket_width = (upper_bound - lower_bound) / class_num

        # Create mask for zero values
        mask_zero = np.ones_like(temp)
        mask_zero[temp == 0] = 0

        np.seterr(divide='ignore', invalid='ignore')

        try:
            # Discretize: assign each value to a bin
            temp = temp.apply(lambda x: 0 if x == 0.0 else math.floor(
                (np.log(x) - lower_bound) / bucket_width))
        except:
            temp = temp.apply(lambda x: 0 if x == 0.0 else 0)

        # Clip values to valid range
        temp[temp >= class_num] = class_num - 1
        temp[(temp < 0)] = 0

        # Shift bins to avoid zero (reserve 0 for actual zeros)
        temp = temp + 1
        temp = temp * mask_zero  # Apply zero mask
        new_data_dict[gene] = temp

    # Convert to tensor
    new_data = pd.DataFrame(new_data_dict)
    new_data = torch.tensor(new_data.T.values, dtype=torch.float32)

    return new_data


























