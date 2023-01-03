import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from torchvision import datasets, transforms
import numpy as np
import torch
import copy

from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid
from models.Nets import CNNMnist
from models.Update import LocalUpdate
from models.test import test_img
from PIL import Image
from models.Fed import FedAvg


if __name__ == '__main__':
    # parse args
    args = args_parser()
    # CUDA: khai thác sức mạnh của GPU cho phần tính toán song song
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        dataset_train, data_validation = torch.utils.data.random_split(dataset_train, [50000, 10000])
        # sample users
        if args.iid:
            # chia data theo IID
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
# thêm
    local_eps=[5,10]
    local_bss=[10,20]
    net_glob_best=CNNMnist(args=args).to(args.device)
    acc_best=0
    args_best=copy.deepcopy(args)
#   
    for local_e in local_eps:
        for local_bs in local_bss:
            args.local_ep=local_e
            args.local_bs=local_bs
            # build model
            if args.model == 'cnn' and args.dataset == 'mnist':
                net_glob = CNNMnist(args=args).to(args.device)
            else:
                exit('Error')
            net_glob.train()

            # copy weights
            w_glob = net_glob.state_dict()
            print("\nTraining with: epochs={}, local_bs={}, local_epochs={}".format(args.epochs, args.local_bs, args.local_ep))
            # training
            loss_train = []
            if args.all_clients: 
                print("Aggregation over all clients")
                w_locals = [w_glob for i in range(args.num_users)]  
            for iter in range(args.epochs):
                loss_locals = []
                if not args.all_clients:
                    w_locals = []
                m = max(int(args.frac * args.num_users), 1)# chọn số client cần đào tạo, tối thiểu là 1
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)# chọn list các user cho training
                for idx in idxs_users:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) # mỗi phần tử của dict_users chứa index data của mỗi client
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    if args.all_clients:
                        w_locals[idx] = copy.deepcopy(w)
                    else:
                        w_locals.append(copy.deepcopy(w))
                    loss_locals.append(copy.deepcopy(loss))
                # update global weights
                w_glob = FedAvg(w_locals)

                # copy weight to net_glob
                net_glob.load_state_dict(w_glob)

                # print loss
                loss_avg = sum(loss_locals) / len(loss_locals)
                print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
                loss_train.append(loss_avg)

            # plot loss curve
            plt.figure()
            plt.plot(range(len(loss_train)), loss_train)
            plt.ylabel('train_loss')
            plt.savefig('./save/fed_{}_{}_glepoch{}_lcbs{}_lcepoch{}.png'.format(args.dataset, args.model, args.epochs, args.local_bs, args.local_ep))

            # testing
            net_glob.eval()
            acc_validation, loss_validation = test_img(net_glob, data_validation, args)
            print("Validation accuracy: {:.2f}".format(acc_validation)) 
            # compare model  
            if acc_best<acc_validation:
                net_glob_best=net_glob
                acc_best=acc_validation
                args_best=copy.deepcopy(args)         
    print("\n============Select Best Model=================")
    acc_test, loss_test = test_img(net_glob_best, dataset_test, args_best)
    print("Best model have: epochs={}, local_bs={}, local_epochs={}".format(args_best.epochs, args_best.local_bs, args_best.local_ep) ) 
    print("Test accuracy: {:.2f}".format(acc_test))  

    #save model
    model_scripted = torch.jit.script(net_glob_best) # Export to TorchScript
    name_model='model_cnn_'+'glEps'+str(args_best.epochs)+'_lcEps'+str(args_best.local_ep)+'_lcBs'+str(args_best.local_bs)+'.pt'
    model_scripted.save(name_model)
    print("Saved model!!!")   



