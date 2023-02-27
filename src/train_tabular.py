import os
import time
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data import  load_folktables_income
from model import Subspace_MLP, LinesLinear, get_weight
from utils import demographic_parity, equal_opportunity, p_rule, plot_distributions, metric_evaluation, seed_everything
from loss import gap_reg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.size"] = 18



logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./")
    parser.add_argument('--dataset', type=str, default="folktables_income")
    parser.add_argument('--model', type=str, default="MLP")
    parser.add_argument('--target_attr', type=str, default="Smiling")
    parser.add_argument('--sensitive_attr', type=str, default="sex")

    parser.add_argument('--epoch', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_hidden', type=int, default=256)

    parser.add_argument('--lam', type=float, default=1)
    parser.add_argument('--gam', type=float, default=20)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--eps', type=float, default=0.05)

    parser.add_argument('--seed', type=int, default=1314)
    parser.add_argument('--exp_name', type=str, default="hhh_torch")
    parser.add_argument('--log_dir', type=str, default="")
    parser.add_argument('--log_screen', type=str, default="True")
    parser.add_argument('--round', type=int, default=0)

    parser.add_argument('--used_mem', type=int, default=2500)
    parser.add_argument('--reg', type=str, default="gap_dp")


    args = parser.parse_args()
    data_path = args.data_path
    dataset_name = args.dataset
    seed = args.seed
    log_screen = eval(args.log_screen)
    log_dir = args.log_dir
    target_attr = args.target_attr
    sensitive_attr = args.sensitive_attr
    exp_name = args.exp_name

    model = args.model
    target_attr = args.target_attr
    num_hidden = args.num_hidden
    reg = args.reg

    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    momentum = args.momentum
    used_mem = args.used_mem

    round = args.round

    lam = args.lam
    gam = args.gam
    beta = args.beta
    eps= args.eps


    seed_everything(seed=seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    used_mem = int( used_mem*0.5 )

    x = torch.zeros(
            (256, 1024, used_mem),
            dtype=torch.float32,
            device=device
        )
    del x


    if log_dir != "":
        log_dir = os.path.join(log_dir, exp_name, dataset_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file_name = f'/{dataset_name}_gmixup_model_{model}_ge_seed_{seed}_round_{round}_{int(time.time())}.log'
        fh = logging.FileHandler(log_dir + log_file_name)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

    if log_screen is True:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.info('parser.prog: {}'.format(parser.prog))
    logger.info("args:{}".format(args))


    if dataset_name == "folktables_income":
        logger.info(f'Dataset: folktables_income')
        X, y, s = load_folktables_income(sensitive_attributes=sensitive_attr)

    else:
        logger.info(f'Wrong dataset_name')

    logger.info(f'X.shape: {X.shape}')
    logger.info(f'y.shape: {y.shape}')
    logger.info(f's.shape: {s.shape}')
    logger.info(f's.shape: {s.value_counts()}')


    n_features = X.shape[1]
    # split into train/val/test set
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, s, test_size=0.2, stratify=y, random_state=1314)
    X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
        X_train, y_train, s_train, test_size=0.1 / 0.8, stratify=y_train, random_state=1314)


    logger.info(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}, s_train.shape: {s_train.shape}')
    logger.info(f'X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}, s_val.shape: {s_val.shape}')
    logger.info(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}, s_test.shape: {s_test.shape}')


    # standardize the data
    scaler = StandardScaler().fit(X_train)

    def scale_df(df, scaler): return pd.DataFrame(scaler.transform(df),columns=df.columns, index=df.index)

    X_train = X_train.pipe(scale_df, scaler)
    X_val = X_val.pipe(scale_df, scaler)
    X_test = X_test.pipe(scale_df, scaler)

    train_data = PandasDataSet(X_train, y_train, s_train)
    val_data = PandasDataSet(X_val, y_val, s_val)
    test_data = PandasDataSet(X_test, y_test, s_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader( val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader( test_data, batch_size=batch_size, shuffle=False)


    clf = Subspace_MLP(n_features=n_features, n_hidden=256).to(device)

    # clf = MLP(n_features=n_features, n_hidden=256).to(device)
    clf_criterion = nn.BCELoss()
    # fair_loss = UnFairnessReduction(gam=20, beta=0.5, eps=0.01)

    # fair_loss = gap_reg()
    if reg == "gap_dp":
        logger.info(f'reg: gap_dp')
        fair_loss = gap_reg(mode="dp")
    elif reg == "gap_eo":
        logger.info(f'reg: gap_eo')
        fair_loss = gap_reg(mode="eo")
    else:
        fair_loss = gap_reg()


    clf_optimizer = optim.Adam(clf.parameters(), lr=0.001)
    clf_optimizer1 = optim.Adam(clf.parameters(), lr=0.001)
    clf_optimizer0 = optim.Adam(clf.parameters(), lr=0.001)

    # N_CLF_EPOCHS = 100

    for epoch in range(num_epochs):


        ce_loss_list = []
        f_loss_list = []
        rate_all_list = []
        rate_1_list = []
        rate_2_list = []
        cosine_list = []

        for x, y, s in train_loader:


            alpha = np.random.uniform(0, 1)
            for m in clf.modules():
                if isinstance(m, nn.Linear):
                    setattr(m, f"alpha", alpha)

            s0 = s
            x = x.to(device)
            y = y.to(device)
            clf.zero_grad()
            p_y_all = clf(x)

            f_loss_all_0, _, _, _ = fair_loss( p_y_all.reshape(-1), s0.reshape(-1), y.reshape(-1))
            loss_all = clf_criterion(p_y_all, y) + alpha*lam*f_loss_all_0


            num_points = 2
            out = random.sample([i for i in range(num_points)], 2)
            i, j = out[0], out[1]
            num = 0.0
            normi = 0.0
            normj = 0.0
            for m in clf.modules():
                if isinstance(m,  LinesLinear):
                    vi = get_weight(m, i)
                    vj = get_weight(m, j)
                    num += (vi * vj).sum()
                    normi += vi.pow(2).sum()
                    normj += vj.pow(2).sum()
            cos_loss = 1 * (num.pow(2) / (normi * normj))
            # loss = loss + cos_loss


            # loss = loss_all + loss_0 + loss_1 + cos_loss
            loss = loss_all + cos_loss

            ce_loss_list.append(loss.item())
            cosine_list.append(cos_loss.item())

            loss.backward()
            clf_optimizer.step()


        epoch_dict = vars(args)
        epoch_dict["epoch"] = epoch
        epoch_dict["ce_loss"] = np.mean(ce_loss_list)
        epoch_dict["f_loss"] = np.mean(f_loss_list)
        epoch_dict["rate_all"] = np.mean(rate_all_list)
        epoch_dict["rate_1"] = np.mean(rate_1_list)
        epoch_dict["rate_2"] = np.mean(rate_2_list)



        # train
        with torch.no_grad():
            y_pre_list = []
            for x, y, s in train_loader:
                x = x.to(device)
                y = y.to(device)
                p_y = clf(x)
                y_pre_list.append(p_y[:, 0].data.cpu().numpy())



        pre_clf_train = np.concatenate(y_pre_list)
        s_train_sex = s_train

        train_metric = metric_evaluation( y_gt=y_train.values, y_pre= pre_clf_train, s=s_train_sex.values, prefix="")
        epoch_dict.update(train_metric)



        # val
        with torch.no_grad():
            y_pre_list = []
            for x, y, s in val_loader:
                x = x.to(device)
                y = y.to(device)
                p_y = clf(x)
                y_pre_list.append(p_y[:, 0].data.cpu().numpy())


        pre_clf_val = np.concatenate(y_pre_list)
        s_val_sex = s_val

        val_metric = metric_evaluation( y_gt=y_val.values, y_pre= pre_clf_val, s=s_val_sex.values, prefix="")
        epoch_dict.update(val_metric)

        # test
        with torch.no_grad():
            y_pre_list = []
            for x, y, s in test_loader:
                x = x.to(device)
                y = y.to(device)
                p_y = clf(x)
                y_pre_list.append(p_y[:, 0].data.cpu().numpy())



        # pre_clf_test = pre_clf_test_all.detach().cpu().numpy()
        pre_clf_test = np.concatenate(y_pre_list)
        s_test_sex = s_test

        train_metric = metric_evaluation( y_gt=y_test.values, y_pre= pre_clf_test, s=s_test_sex.values, prefix="test_")
        epoch_dict.update(train_metric)
        
        #alpha
        with torch.no_grad():
            alpha_dict = dict()
            for alpha in np.linspace( 0, 1, num=21):
                for m in clf.modules():
                    if isinstance(m, nn.Linear):
                        setattr(m, f"alpha", alpha)

                # test
                with torch.no_grad():
                    y_pre_list = []
                    for x, y, s in test_loader:
                        x = x.to(device)
                        y = y.to(device)
                        p_y = clf(x)
                        y_pre_list.append(p_y[:, 0].data.cpu().numpy())
            
                pre_clf_test = np.concatenate(y_pre_list)
                s_test_sex = s_test

                train_metric = metric_evaluation( y_gt=y_test.values, y_pre= pre_clf_test, s=s_test_sex.values, prefix="test_")
                alpha_dict[alpha] = train_metric


        epoch_dict[ "alpha_dict" ] = alpha_dict

        import pandas as pd
        res =   pd.DataFrame( alpha_dict ).T


        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["font.size"] = 18


        t = np.arange(0, 1, 0.01)
        data1 = np.exp(t)
        data2 = np.sin(2 * np.pi * t)

        t = np.linspace(0, 1, 21)

        data1 = 100 - res["test_ap"].values
        data2 = res["test_dpe"].values * 100

        data1 = data1[::-1]
        data2 = data2[::-1]


        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('$\\alpha$')
        ax1.set_ylabel('Error Rate', color=color)
        ax1.plot(t, data1, color=color, marker = "o")
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('$\Delta \mathrm{DP}_{\mathrm{gender}}$', color=color)  # we already handled the x-label with ax1
        ax2.plot(t, data2, color=color, marker = "o")
        ax2.tick_params(axis='y', labelcolor=color)

        ax1.grid()

        fig.tight_layout() 
        plt.savefig( f"./alpha.pdf", bbox_inches='tight' )
        plt.show()

    logger.info(f"done experiment")
