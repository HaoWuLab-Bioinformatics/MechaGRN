import argparse
import os
import random
import time
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from Tools import Evaluation, SavaBestModel, write_to_txt
from preprocessing import scRNADataset, load_data, adj2saprse_tensor

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings(action='ignore', category=FutureWarning)

# ======================== Hyperparameters ========================
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for optimizer')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension size')
parser.add_argument('--output_dim', type=int, default=16, help='Output embedding dimension')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
parser.add_argument('--loop', type=bool, default=False, help='Whether to add self-loops to adjacency matrix')
parser.add_argument('--seed', type=int, default=40, help='Random seed for reproducibility')
args = parser.parse_args()

# ======================== Fix Random Seeds ========================
# Ensures reproducibility across different runs
seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

# ======================== Run Model ========================

def Running_Model(net_type, cell_type, gene_num, record_model):
    """
    Train and evaluate MechaGRN model on specified dataset.

    Args:
        net_type: Network type (Specific, Non-Specific, STRING, Lofgof)
        cell_type: Cell type identifier (e.g., hESC, mESC, mHSC-GM)
        gene_num: Number of genes in the network (500 or 1000)
        record_model: Whether to print model architecture at the end
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ======================== File Paths ========================
    # Expression data and TF indices
    exp_file = f'../Dataset/Benchmark Dataset/{net_type} Dataset/{cell_type}/TFs+{gene_num}/BL--ExpressionData.csv'
    tf_file = f'../Dataset/Benchmark Dataset/{net_type} Dataset/{cell_type}/TFs+{gene_num}/TF.csv'

    # Train, validation, and test set paths
    train_file = f'../Dataset/train/{net_type}/{cell_type} {gene_num}/Train_set.csv'
    val_file = f'../Dataset/val/{net_type}/{cell_type} {gene_num}/Validation_set.csv'
    test_file = f'../Dataset/test/{net_type}/{cell_type} {gene_num}/test_set.csv'

    # ======================== Load Expression Matrix ========================
    # Read gene expression data and normalize
    data_input = pd.read_csv(exp_file, index_col=0)
    loader = load_data(data_input)
    feature = loader.exp_data()
    feature = torch.tensor(feature, dtype=torch.float32).to(device)

    # ======================== Load TF Indices ========================
    # Read transcription factor gene indices
    tf = pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int64)
    tf = torch.tensor(tf)

    # ======================== Build Adjacency Matrix ========================
    # Construct sparse adjacency matrix from training links
    train_data = pd.read_csv(train_file, index_col=0).values
    train_load = scRNADataset(train_data, feature.shape[0])
    adj = train_load.Adj_Generate(tf, loop=args.loop)
    adj = adj2saprse_tensor(adj).to(device)

    # ======================== Validation & Test Data ========================
    # Load validation and test samples as tensors
    val_data = torch.tensor(pd.read_csv(val_file, index_col=0).values).to(device)
    test_data = torch.tensor(pd.read_csv(test_file, index_col=0).values).to(device)

    # ======================== Initialize Model ========================
    # Create MechaGRN with specified hyperparameters
    from Model import MechaGRN, FocalLoss

    model = MechaGRN(
        input_dim=feature.shape[1],   # Number of cells (expression matrix columns)
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_heads=args.num_heads,
        device=device,
        use_expression=True
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    # Use Focal Loss to handle class imbalance in regulatory links
    loss_focal = FocalLoss(alpha=0.8, gamma=2.5)

    # Create directory for saving best model
    model_path = f'model/{net_type}/{cell_type} {gene_num}'
    os.makedirs(model_path, exist_ok=True)
    best_model = SavaBestModel(True, model_path)

    print(f"Cell: {cell_type} | Network: {net_type} | Genes: {gene_num}")
    print("=========== Start Training ==========")

    # ======================== Training Loop ========================
    for epoch in range(args.epochs):

        model.train()
        running_loss = 0.0

        data_loader = DataLoader(train_load, batch_size=args.batch_size, shuffle=True)

        for train_x, train_y in tqdm(data_loader):

            train_x = train_x.to(device)
            train_y = train_y.to(device).view(-1, 1)

            optimizer.zero_grad()

            pred = model(feature, adj, train_x)
            loss = loss_focal(pred, train_y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # ======================== Validation ========================
        # Evaluate on validation set and save best model
        model.eval()
        with torch.no_grad():
            score_val = model(feature, adj, val_data)
            score_val = torch.sigmoid(score_val)

            AUROC, AUPRC, F1, EP = Evaluation(
                y_pred=score_val,
                y_true=val_data[:, -1]
            )

            best_model(AUROC, AUPRC, model)

        print(f"Epoch [{epoch+1}/{args.epochs}] | Loss: {running_loss:.2f} | AUROC: {AUROC:.2f} | AUPRC: {AUPRC:.2f} | F1: {F1:.2f} | EP: {EP:.2f}")

    # ======================== Testing ========================
    # Load best model and evaluate on test set
    print("=========== Testing ==========")

    model.load_state_dict(torch.load(model_path + '/best_model.pkl'))
    model.eval()

    with torch.no_grad():
        # Apply temperature scaling for calibrated predictions
        score_test = model(feature, adj, test_data) / 0.8
        score_test = torch.clamp(score_test, -6, 6)  # Clip extreme values
        score_test = torch.sigmoid(score_test)

        AUROC, AUPRC, F1, EP = Evaluation(
            y_pred=score_test,
            y_true=test_data[:, -1]
        )

    print(f"Test AUROC: {AUROC:.2f} | AUPRC: {AUPRC:.2f} | F1: {F1:.2f} | EP: {EP:.2f}")
    write_to_txt(net_type, cell_type, gene_num, AUROC, AUPRC, F1)
    if record_model:
        print(model)


# ======================== Main Entry Point ========================

if __name__ == '__main__':
    # Configure network types and cell types for experiments
    net_types = ["Specific"]
    # Available cell types: "hESC", "hHEP", "mDC", "mESC", "mHSC-E", "mHSC-GM", "mHSC-L"
    cell_types = ["mHSC-GM"]
    gene_num = 500  # Number of genes in the network (500 or 1000)

    start_time = time.time()

    for net_type in net_types:
        for i, cell_type in enumerate(cell_types):
            record_model = True if i == len(cell_types)-1 else False
            Running_Model(net_type, cell_type, gene_num, record_model)


    end_time = time.time()
    print(f"Total time: {(end_time - start_time):.2f} seconds")