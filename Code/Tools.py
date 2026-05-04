import torch
import numpy as np
import sys
import time
from sklearn.metrics import roc_auc_score, recall_score, average_precision_score, f1_score
from sklearn.metrics import precision_recall_curve



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_dir, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.

                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.

                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.

                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.save_dir = save_dir
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.

        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')
        torch.save(model.state_dict(), self.save_dir + '.pkl')
        self.val_loss_min = val_loss

class SavaBestModel:

    def __init__(self,trigger,save_dir):
        self.Trigger = trigger
        self.save_dir = save_dir
        self.best_score = None

    def __call__(self, auroc, auprc, model):
        if self.Trigger:
            score = 0.2 * auroc + 0.8 * auprc
        else:
            score = 999999999999999999999

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)

        elif score < self.best_score:
            pass

        else:
            self.best_score = score
            self.save_checkpoint(model)

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_dir+'/best_model.pkl')



def progress_bar(finish_tasks_number, tasks_number):
    """
    进度条

    :param finish_tasks_number: int, 已完成的任务数
    :param tasks_number: int, 总的任务数
    :return:
    """

    percentage = round(finish_tasks_number / tasks_number * 100)
    print("\r进度: {}%: ".format(percentage), "▓" * (percentage // 2), end="")
    sys.stdout.flush()
def early_precision(y_true, y_score, recall_threshold=0.1):
    """
    计算 Early Precision (EP)

    参数
    ----------
    y_true : array-like
        真实标签 (0/1)
    y_score : array-like
        模型预测得分
    recall_threshold : float
        early region 的 recall 上限 (默认 0.1)

    返回
    ----------
    ep : float
        Early Precision
    """

    precision, recall, _ = precision_recall_curve(y_true, y_score)

    # 只保留 recall <= threshold 的部分
    mask = recall <= recall_threshold

    if np.sum(mask) == 0:
        return 0

    ep = np.mean(precision[mask])

    return ep
def Evaluation(y_true, y_pred,flag=False):
    if flag:
        y_p = y_pred[:,-1]
        y_p = y_p.cpu().detach().numpy()
        y_p = y_p.flatten()
    else:
        y_p = y_pred.cpu().detach().numpy()
        y_p = y_p.flatten()

    y_t = y_true.cpu().numpy().flatten().astype(int)
    AUC = roc_auc_score(y_true=y_t, y_score=y_p)
    AUPR = average_precision_score(y_true=y_t,y_score=y_p)
    y_pred_label = (y_p > 0.5).astype(int)
    F1_score = f1_score(y_true=y_t,y_pred=y_pred_label)
    EP = early_precision(y_t, y_p)
    # Recall_score = recall_score(y_true=y_t,y_score=y_p)

    return AUC, AUPR, F1_score, EP

def write_to_txt(net_type, cell_type, gene_num, auroc, auprc, f1, filename='results.txt'):
    with open(filename, mode='a') as file:
        # 写入表头（如果文件为空）
        if file.tell() == 0:
            file.write("实验结果记录V2\n")
            file.write("=======================================")
        file.write(f"\n{net_type}\t{cell_type}\t{gene_num}\t{auroc:.4f}\t{auprc:.4f}\t{f1:.4f}")



if __name__ == '__main__':
    for i in range(0, 101):
        progress_bar(i, 100)
        time.sleep(0.05)

