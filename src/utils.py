
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import os
import torch
from sklearn import metrics
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def demographic_parity_auc(y_pred, y_gt, z_values):
    y_pred = y_pred.ravel()
    y_gt = y_gt.ravel()
    z_values = z_values.ravel()
    # print( y_pred.shape, y_gt.shape, z_values.shape )

    y_pre_1 = y_pred[z_values == 1]
    y_pre_0 = y_pred[z_values == 0]

    y_gt_1 = y_gt[z_values == 1]
    y_gt_0 = y_gt[z_values == 0]

    auc1 = metrics.roc_auc_score( y_true=y_gt_1, y_score= y_pre_1)
    auc0 = metrics.roc_auc_score( y_true=y_gt_0, y_score= y_pre_0)  

    auc_parity = abs(auc1 - auc0)
    return auc_parity
    # return 0


def demographic_parity(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    parity = abs(y_z_1.mean() - y_z_0.mean())
    return parity


def equal_opportunity(y_pred, y_gt, z_values, threshold=0.5):
    y_pred = y_pred[y_gt == 1]  # y_gt == 1
    z_values = z_values[y_gt == 1]
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    equality = abs(y_z_1.mean() - y_z_0.mean())
    return equality


def p_rule(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    return np.nanmin([odds, 1/odds]) * 100


def plot_distributions(y_pred, s, Z_pred=None, epoch=None):

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharey=True)


    df = pd.DataFrame( s.copy() )
    sensitive_attribute = list( df.columns)
    df["y_pred"] = y_pred

    # print( df )
    # print(sensitive_attribute)
    
    for label, df in df.groupby(sensitive_attribute):
        # sns.kdeplot(df['y_pred'], ax=ax, label=label, shade=True)
        sns.distplot(df['y_pred'], ax=ax, label=label, shade=True)
        
    # sns.kdeplot(df['y_pred'], ax=ax, label=label, shade=True)

    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


def seed_everything(seed=1314):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def metric_evaluation(y_gt, y_pre, s, prefix):
    
    y_gt = y_gt.ravel()
    y_pre = y_pre.ravel()
    s= s.ravel()

    accuracy = metrics.accuracy_score(y_gt, y_pre > 0.5) * 100
    ap = metrics.average_precision_score(y_gt, y_pre) * 100
    dp = demographic_parity(y_pre, s)
    dpe = demographic_parity(y_pre, s, threshold=None)
    eo = equal_opportunity(y_pre, y_gt, s)
    eoe = equal_opportunity(y_pre, y_gt, s, threshold=None)
    prule = p_rule(y_pre, s)
    prulee = p_rule(y_pre, s, threshold=None)

    metric_name = [ "accuracy", "ap", "dp", "dpe", "eo", "eoe", "prule", "prulee" ]
    metric_name = [ prefix+x for x in metric_name]
    metric_val = [ accuracy, ap, dp, dpe, eo, eoe, prule, prulee ]

    return dict( zip( metric_name, metric_val))


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()