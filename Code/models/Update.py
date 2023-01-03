import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics

# phân vùng cho tập train và validate, chứa list id của dataset {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
# dictionary của nhãn (lable)   {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.idxs = list(idxs)
        self.dataset = dataset

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        #CrossEntropyLoss: Tiêu chí này tính toán tổn thất entropy chéo giữa đầu vào và mục tiêu
        self.loss_function = nn.CrossEntropyLoss()
        self.select_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs),  shuffle=True ,batch_size=self.args.local_bs)

    def train(self, net):
        # train and update
        net.train()
        # Thực hiện giảm độ dốc ngẫu nhiên (tùy chọn với động lượng(momentum))
        optimizer = torch.optim.SGD(net.parameters(),  momentum=self.args.momentum,lr=self.args.lr)
        loss_epoch = []
        for epoch in range(self.args.local_ep):
            loss_batch = []
            for idx_batch, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                lossF = self.loss_function(log_probs, labels)
                lossF.backward()
                optimizer.step()
                if self.args.verbose and idx_batch % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, idx_batch * len(images), len(self.ldr_train.dataset),
                               100. * idx_batch / len(self.ldr_train), lossF.item()))
                loss_batch.append(lossF.item())
            loss_epoch.append(sum(loss_batch)/len(loss_batch))
        return net.state_dict(), sum(loss_epoch) / len(loss_epoch)

