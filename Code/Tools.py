import torch
import numpy as np
import sys
import time
from sklearn.metrics import roc_auc_score, recall_score, average_precision_score, f1_score
from sklearn.metrics import precision_recall_curve


class EarlyStopping:
    """
    Early stopping callback to stop training when validation loss stops improving.

    Args:
        save_dir: Directory path for saving model checkpoint
        patience: Number of epochs to wait before stopping (default: 7)
        verbose: Whether to print improvement messages (default: False)
        delta: Minimum change to qualify as an improvement (default: 0)
    """

    def __init__(self, save_dir, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.save_dir = save_dir
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        """Check validation loss and update early stopping state."""
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # No improvement
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement found
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Save model checkpoint when validation loss decreases.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_dir + '.pkl')
        self.val_loss_min = val_loss


class SavaBestModel:
    """
    Callback to save the best model based on AUROC and AUPRC metrics.

    Args:
        trigger: Whether to save based on metrics (True) or always skip (False)
        save_dir: Directory path for saving best model
    """

    def __init__(self, trigger, save_dir):
        self.Trigger = trigger
        self.save_dir = save_dir
        self.best_score = None

    def __call__(self, auroc, auprc, model):
        """
        Check metrics and save model if improvement is found.
        Uses combined score: 0.2 * AUROC + 0.8 * AUPRC
        """
        if self.Trigger:
            score = 0.2 * auroc + 0.8 * auprc
        else:
            score = 999999999999999999999  # Never save if trigger is False

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)

        elif score < self.best_score:
            pass  # No improvement

        else:
            # Improvement found
            self.best_score = score
            self.save_checkpoint(model)

    def save_checkpoint(self, model):
        """Save model state dict to specified directory."""
        torch.save(model.state_dict(), self.save_dir + '/best_model.pkl')


def progress_bar(finish_tasks_number, tasks_number):
    """
    Display a simple text progress bar.

    Args:
        finish_tasks_number: Number of completed tasks
        tasks_number: Total number of tasks
    """

    percentage = round(finish_tasks_number / tasks_number * 100)
    print("\rProgress: {}%: ".format(percentage), "▓" * (percentage // 2), end="")
    sys.stdout.flush()


def early_precision(y_true, y_score, recall_threshold=0.1):
    """
    Calculate Early Precision (EP) metric.

    Early precision measures the precision in the early region of the
    precision-recall curve, where recall is below the threshold.
    This is important for GRN inference where identifying true links
    early in the ranking is valuable.

    Args:
        y_true: True binary labels (0/1)
        y_score: Predicted scores/probabilities
        recall_threshold: Recall upper bound for early region (default: 0.1)

    Returns:
        ep: Early Precision value (mean precision in early region)
    """

    precision, recall, _ = precision_recall_curve(y_true, y_score)

    # Keep only points where recall <= threshold
    mask = recall <= recall_threshold

    if np.sum(mask) == 0:
        return 0

    ep = np.mean(precision[mask])

    return ep


def Evaluation(y_true, y_pred, flag=False):
    """
    Evaluate model predictions using multiple metrics.

    Args:
        y_true: True binary labels
        y_pred: Model predictions (logits or probabilities)
        flag: Whether predictions are from a multi-class output (default: False)

    Returns:
        AUC: Area Under ROC Curve
        AUPR: Area Under Precision-Recall Curve
        F1_score: F1 score at threshold 0.5
        EP: Early Precision
    """
    if flag:
        # Extract probability from last column for multi-class output
        y_p = y_pred[:, -1]
        y_p = y_p.cpu().detach().numpy()
        y_p = y_p.flatten()
    else:
        y_p = y_pred.cpu().detach().numpy()
        y_p = y_p.flatten()

    y_t = y_true.cpu().numpy().flatten().astype(int)

    # Calculate metrics
    AUC = roc_auc_score(y_true=y_t, y_score=y_p)
    AUPR = average_precision_score(y_true=y_t, y_score=y_p)

    # Binary predictions at threshold 0.5
    y_pred_label = (y_p > 0.5).astype(int)
    F1_score = f1_score(y_true=y_t, y_pred=y_pred_label)

    EP = early_precision(y_t, y_p)

    return AUC, AUPR, F1_score, EP


def write_to_txt(net_type, cell_type, gene_num, auroc, auprc, f1, filename='results.txt'):
    """
    Write experiment results to a text file.

    Args:
        net_type: Network type name
        cell_type: Cell type name
        gene_num: Number of genes
        auroc: AUROC score
        auprc: AUPRC score
        f1: F1 score
        filename: Output file name (default: 'results.txt')
    """
    with open(filename, mode='a') as file:
        # Write header if file is empty
        if file.tell() == 0:
            file.write("Experiment Results Record V2\n")
            file.write("=======================================")
        file.write(f"\n{net_type}\t{cell_type}\t{gene_num}\t{auroc:.4f}\t{auprc:.4f}\t{f1:.4f}")


if __name__ == '__main__':
    # Demo: progress bar visualization
    for i in range(0, 101):
        progress_bar(i, 100)
        time.sleep(0.05)