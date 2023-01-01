from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    loss_of_test = 0
    num_correct = 0
    dataLoad = DataLoader(datatest, batch_size=args.bs)
    l = len(dataLoad)
    for idx, (data, target) in enumerate(dataLoad):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        prob_log = net_g(data)
        # sum up batch loss
        loss_of_test += F.cross_entropy(prob_log, target, reduction='sum').item()
        # get the index of the max log-probability
        y_predict = prob_log.data.max(1, keepdim=True)[1]
        num_correct += y_predict.eq(target.data.view_as(y_predict)).long().cpu().sum()

    loss_of_test /= len(dataLoad.dataset)
    accuracy = 100.00 * num_correct / len(dataLoad.dataset)
    # print detail of loss when args.verbose = true
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            loss_of_test, num_correct, len(dataLoad.dataset), accuracy))
    return accuracy, loss_of_test

