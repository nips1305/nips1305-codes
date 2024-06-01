import os
import torch
import warnings
import numpy as np
from model import GCGQ
from data_loader import load_data

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    seed = 114514
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    name = 'cora'

    method_name = 'GCGQ'
    features, adjacency, labels = load_data(name)
    layers = [512, 256, 128]
    acts = [torch.nn.functional.relu] * len(layers)
    learning_rate = 10 ** -4 * 4
    pretrain_learning_rate = 0.0001
    lamSC = np.power(2.0, -10)
    coeff_reg = 0.00001

    max_epoch = 50
    max_iter = 4
    pre_iter = 10


    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []

    for _ in range(10):
        gae = GCGQ(name, features, adjacency, labels, layers=layers,
                   acts=acts, max_epoch=max_epoch, max_iter=max_iter, coeff_reg=coeff_reg, learning_rate=learning_rate,
                   seed=seed, lam=lamSC)

        gae.cuda()

        gae.pretrain(pre_iter, learning_rate=pretrain_learning_rate)
        acc, nmi, ari, f1 = gae.run()

        acc_list.append(acc)
        nmi_list.append(nmi)
        ari_list.append(ari)
        f1_list.append(f1)

    print("\n")
    acc_list = np.array(acc_list)
    nmi_list = np.array(nmi_list)
    ari_list = np.array(ari_list)
    f1_list = np.array(f1_list)

    print(acc_list.mean(), "±", acc_list.std())
    print(nmi_list.mean(), "±", nmi_list.std())
    print(ari_list.mean(), "±", ari_list.std())
    print(f1_list.mean(), "±", f1_list.std())


