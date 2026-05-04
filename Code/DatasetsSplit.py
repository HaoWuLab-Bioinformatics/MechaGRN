import pandas as pd
import numpy as np
import os
from preprocessing import Network_Statistic
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', type=float, default=0.67, help='Ratio of training set (default: 0.67)')
parser.add_argument('--p_val', type=float, default=0.5, help='Probability for single-target TFs to go to training set (default: 0.5)')
args = parser.parse_args()


def train_val_test_set(label_file, Gene_file, TF_file, train_set_file, val_set_file, test_set_file, density, net_type, data_type, gene_num, p_val=args.p_val):
    """
    Split dataset into train, validation, and test sets for non-specific networks.

    For each TF, splits its targets into training and test sets.
    Uses random sampling for negative examples.

    Args:
        label_file: Path to ground truth regulatory links
        Gene_file: Path to target gene indices
        TF_file: Path to TF indices
        train_set_file: Output path for training set
        val_set_file: Output path for validation set
        test_set_file: Output path for test set
        density: Network density for negative sampling
        net_type: Network type name
        data_type: Cell type name
        gene_num: Number of genes
        p_val: Probability for single-target TFs to go to training set
    """
    print("================= Starting Dataset Split =================")
    print(f"{net_type}\t{data_type}\t{gene_num}")

    # Load gene and TF sets
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values

    # Load ground truth regulatory links
    label = pd.read_csv(label_file, index_col=0)
    tf = label['TF'].values

    # Build positive link dictionary: TF -> [targets]
    tf_list = np.unique(tf)
    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    for i, j in label.values:
        pos_dict[i].append(j)

    # Split positive links for each TF
    train_pos = {}
    val_pos = {}
    test_pos = {}

    for k in pos_dict.keys():
        if len(pos_dict[k]) <= 1:
            # Single target: randomly assign to train or test
            p = np.random.uniform(0, 1)
            if p <= p_val:
                train_pos[k] = pos_dict[k]
            else:
                test_pos[k] = pos_dict[k]

        elif len(pos_dict[k]) == 2:
            # Two targets: one for train, one for test
            train_pos[k] = [pos_dict[k][0]]
            test_pos[k] = [pos_dict[k][1]]

        else:
            # Multiple targets: split into train/val/test
            np.random.shuffle(pos_dict[k])
            train_pos[k] = pos_dict[k][:len(pos_dict[k]) * 2 // 3]
            test_pos[k] = pos_dict[k][len(pos_dict[k]) * 2 // 3:]

            # Validation set from training portion
            val_pos[k] = train_pos[k][:len(train_pos[k]) // 5]
            train_pos[k] = train_pos[k][len(train_pos[k]) // 5:]

    # Generate negative examples for training
    train_neg = {}
    for k in train_pos.keys():
        train_neg[k] = []
        for i in range(len(train_pos[k])):
            neg = np.random.choice(gene_set)
            # Ensure negative is not TF itself, not a true target, and not duplicate
            while neg == k or neg in pos_dict[k] or neg in train_neg[k]:
                neg = np.random.choice(gene_set)
            train_neg[k].append(neg)

    # Build training set with labels
    train_pos_set = []
    train_neg_set = []
    for k in train_pos.keys():
        for j in train_pos[k]:
            train_pos_set.append([k, j])
    tran_pos_label = [1 for _ in range(len(train_pos_set))]

    for k in train_neg.keys():
        for j in train_neg[k]:
            train_neg_set.append([k, j])
    tran_neg_label = [0 for _ in range(len(train_neg_set))]

    train_set = train_pos_set + train_neg_set
    train_label = tran_pos_label + tran_neg_label

    train_sample = train_set.copy()
    for i, val in enumerate(train_sample):
        val.append(train_label[i])
    train = pd.DataFrame(train_sample, columns=['TF', 'Target', 'Label'])

    train.to_csv(train_set_file)
    print('================= Training Set Complete =================')

    # Build validation set
    val_pos_set = []
    for k in val_pos.keys():
        for j in val_pos[k]:
            val_pos_set.append([k, j])
    val_pos_label = [1 for _ in range(len(val_pos_set))]

    # Generate negative examples for validation
    val_neg = {}
    for k in val_pos.keys():
        val_neg[k] = []
        for i in range(len(val_pos[k])):
            neg = np.random.choice(gene_set)
            while neg == k or neg in pos_dict[k] or neg in train_neg[k] or neg in val_neg[k]:
                neg = np.random.choice(gene_set)
            val_neg[k].append(neg)

    val_neg_set = []
    for k in val_neg.keys():
        for j in val_neg[k]:
            val_neg_set.append([k, j])

    val_neg_label = [0 for _ in range(len(val_neg_set))]
    val_set = val_pos_set + val_neg_set
    val_set_label = val_pos_label + val_neg_label

    val_set_a = np.array(val_set)
    val_sample = pd.DataFrame()
    val_sample['TF'] = val_set_a[:, 0]
    val_sample['Target'] = val_set_a[:, 1]
    val_sample['Label'] = val_set_label
    val_sample.to_csv(val_set_file)
    print('================= Validation Set Complete =================')

    # Build test set
    test_pos_set = []
    for k in test_pos.keys():
        for j in test_pos[k]:
            test_pos_set.append([k, j])

    # Calculate number of negative examples based on density
    count = 0
    for k in test_pos.keys():
        count += len(test_pos[k])
    test_neg_num = int(count // density - count)

    # Generate negative examples for test set
    test_neg = {}
    for k in tf_set:
        test_neg[k] = []

    test_neg_set = []
    for i in range(test_neg_num):
        t1 = np.random.choice(tf_set)
        t2 = np.random.choice(gene_set)
        # Ensure negative is not in training, test positive, or validation
        while t1 == t2 or [t1, t2] in train_set or [t1, t2] in test_pos_set or [t1, t2] in val_set or [t1, t2] in test_neg_set:
            t2 = np.random.choice(gene_set)

        test_neg_set.append([t1, t2])

    test_pos_label = [1 for _ in range(len(test_pos_set))]
    test_neg_label = [0 for _ in range(len(test_neg_set))]

    test_set = test_pos_set + test_neg_set
    test_label = test_pos_label + test_neg_label
    for i, val in enumerate(test_set):
        val.append(test_label[i])

    test_sample = pd.DataFrame(test_set, columns=['TF', 'Target', 'Label'])
    test_sample.to_csv(test_set_file)
    print('================= Test Set Complete =================')


def Hard_Negative_Specific_train_test_val(label_file, Gene_file, TF_file, train_set_file, val_set_file, test_set_file, net_type, data_type, gene_num,
                                          ratio=args.ratio, p_val=args.p_val):
    """
    Split dataset for specific networks with hard negative sampling.

    Hard negatives are genes that are NOT regulated by a TF, providing
    a more challenging negative set for training.

    Args:
        label_file: Path to ground truth regulatory links
        Gene_file: Path to target gene indices
        TF_file: Path to TF indices
        train_set_file: Output path for training set
        val_set_file: Output path for validation set
        test_set_file: Output path for test set
        net_type: Network type name
        data_type: Cell type name
        gene_num: Number of genes
        ratio: Training set ratio (default: 0.67)
        p_val: Probability for single-target TFs (default: 0.5)
    """
    print("================= Starting Dataset Split =================")

    # Load ground truth and gene sets
    label = pd.read_csv(label_file, index_col=0)
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values

    tf = label['TF'].values
    tf_list = np.unique(tf)

    # Build positive link dictionary
    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    for i, j in label.values:
        pos_dict[i].append(j)

    # Build hard negative dictionary
    # Hard negatives: genes NOT regulated by this TF
    neg_dict = {}
    for i in tf_set:
        neg_dict[i] = []

    for i in tf_set:
        if i in pos_dict.keys():
            # TF has known targets: exclude TF itself and targets
            pos_item = pos_dict[i]
            pos_item.append(i)  # Exclude TF itself
            neg_item = np.setdiff1d(gene_set, pos_item)
            neg_dict[i].extend(neg_item)
            pos_dict[i] = np.setdiff1d(pos_dict[i], i)  # Remove TF from targets
        else:
            # TF has no known targets: exclude only TF itself
            neg_item = np.setdiff1d(gene_set, i)
            neg_dict[i].extend(neg_item)

    # Split positive links for each TF
    train_pos = {}
    val_pos = {}
    test_pos = {}
    for k in pos_dict.keys():
        if len(pos_dict[k]) == 1:
            # Single target: randomly assign
            p = np.random.uniform(0, 1)
            if p <= p_val:
                train_pos[k] = pos_dict[k]
            else:
                test_pos[k] = pos_dict[k]

        elif len(pos_dict[k]) == 2:
            # Two targets: split evenly
            np.random.shuffle(pos_dict[k])
            train_pos[k] = [pos_dict[k][0]]
            test_pos[k] = [pos_dict[k][1]]
        else:
            # Multiple targets: split by ratio
            np.random.shuffle(pos_dict[k])
            train_pos[k] = pos_dict[k][:int(len(pos_dict[k]) * ratio)]
            val_pos[k] = pos_dict[k][int(len(pos_dict[k]) * ratio):int(len(pos_dict[k]) * (ratio + 0.1))]
            test_pos[k] = pos_dict[k][int(len(pos_dict[k]) * (ratio + 0.1)):]

    # Split negative examples for each TF
    train_neg = {}
    val_neg = {}
    test_neg = {}
    for k in pos_dict.keys():
        neg_num = len(neg_dict[k])
        np.random.shuffle(neg_dict[k])
        train_neg[k] = neg_dict[k][:int(neg_num * ratio)]
        val_neg[k] = neg_dict[k][int(neg_num * ratio):int(neg_num * (0.1 + ratio))]
        test_neg[k] = neg_dict[k][int(neg_num * (0.1 + ratio)):]

    # Build training set
    train_pos_set = []
    for k in train_pos.keys():
        for val in train_pos[k]:
            train_pos_set.append([k, val])

    train_neg_set = []
    for k in train_neg.keys():
        for val in train_neg[k]:
            train_neg_set.append([k, val])

    train_set = train_pos_set + train_neg_set
    train_label = [1 for _ in range(len(train_pos_set))] + [0 for _ in range(len(train_neg_set))]

    train_sample = np.array(train_set)
    train = pd.DataFrame()
    train['TF'] = train_sample[:, 0]
    train['Target'] = train_sample[:, 1]
    train['Label'] = train_label
    train.to_csv(train_set_file)
    print('================= Training Set Complete =================')

    # Build validation set
    val_pos_set = []
    for k in val_pos.keys():
        for val in val_pos[k]:
            val_pos_set.append([k, val])

    val_neg_set = []
    for k in val_neg.keys():
        for val in val_neg[k]:
            val_neg_set.append([k, val])

    val_set = val_pos_set + val_neg_set
    val_label = [1 for _ in range(len(val_pos_set))] + [0 for _ in range(len(val_neg_set))]

    val_sample = np.array(val_set)
    val = pd.DataFrame()
    val['TF'] = val_sample[:, 0]
    val['Target'] = val_sample[:, 1]
    val['Label'] = val_label
    val.to_csv(val_set_file)
    print('================= Validation Set Complete =================')

    # Build test set
    test_pos_set = []
    for k in test_pos.keys():
        for j in test_pos[k]:
            test_pos_set.append([k, j])

    test_neg_set = []
    for k in test_neg.keys():
        for j in test_neg[k]:
            test_neg_set.append([k, j])

    test_set = test_pos_set + test_neg_set
    test_label = [1 for _ in range(len(test_pos_set))] + [0 for _ in range(len(test_neg_set))]

    test_sample = np.array(test_set)
    test = pd.DataFrame()
    test['TF'] = test_sample[:, 0]
    test['Target'] = test_sample[:, 1]
    test['Label'] = test_label
    test.to_csv(test_set_file)
    print('================= Test Set Complete =================')


if __name__ == '__main__':
    # Configure experiment settings
    net_types = ["Lofgof"]
    data_types = ["mESC"]
    gene_num = 1000

    for net_type in net_types:
        for data_type in data_types:
            # Get network density for test set sampling
            density = Network_Statistic(data_type=data_type, net_scale=gene_num, net_type=net_type)

            # Define file paths
            TF2file = "../Dataset/Benchmark Dataset" + '/' + net_type + ' Dataset/' + data_type + '/TFs+' + str(gene_num) + '/TF.csv'
            Gene2file = "../Dataset/Benchmark Dataset" + '/' + net_type + ' Dataset/' + data_type + '/TFs+' + str(gene_num) + '/Target.csv'
            label_file = "../Dataset/Benchmark Dataset" + '/' + net_type + ' Dataset/' + data_type + '/TFs+' + str(gene_num) + '/Label.csv'

            # Create output directories if not exist
            train_set_file = "../Dataset" + '/train/' + net_type + '/' + data_type + ' ' + str(gene_num) + '/'
            if not os.path.exists(train_set_file):
                os.makedirs(train_set_file)
            train_set_file = os.path.join(train_set_file, 'Train_set.csv')

            test_set_file = "../Dataset" + '/test/' + net_type + '/' + data_type + ' ' + str(gene_num) + '/'
            if not os.path.exists(test_set_file):
                os.makedirs(test_set_file)
            test_set_file = os.path.join(test_set_file, 'Test_set.csv')

            val_set_file = "../Dataset" + '/val/' + net_type + '/' + data_type + ' ' + str(gene_num) + '/'
            if not os.path.exists(val_set_file):
                os.makedirs(val_set_file)
            val_set_file = os.path.join(val_set_file, 'Validation_set.csv')

            # Use hard negative sampling for Specific networks
            if net_type == 'Specific':
                Hard_Negative_Specific_train_test_val(label_file, Gene2file, TF2file, train_set_file, val_set_file,
                                                      test_set_file, net_type, data_type, gene_num)
            else:
                train_val_test_set(label_file, Gene2file, TF2file, train_set_file, val_set_file, test_set_file, density, net_type, data_type, gene_num)