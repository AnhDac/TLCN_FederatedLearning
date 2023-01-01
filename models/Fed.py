import copy
import torch
from torch import nn


def FedAvg(w):
    lst_w_avg = copy.deepcopy(w[0])
    for client in lst_w_avg.keys():
        for i in range(1, len(w)):
            lst_w_avg[client] += w[i][client]
        lst_w_avg[client] = torch.div(lst_w_avg[client], len(w))
    return lst_w_avg
