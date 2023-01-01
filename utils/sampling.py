import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, number_users):
    # chia data ngẫu nhiên cho K người tham gia
    number_items = int(len(dataset)/number_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for num in range(number_users):
        dict_users[num] = set(np.random.choice(all_idxs, number_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[num])
    return dict_users


def mnist_noniid(dataset, number_users):
    number_shards, num_imgs = 200, 300
    idx_shard_lst = [n for n in range(number_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(number_users)}
    idxs = np.arange(number_shards*num_imgs)
    lst_lable = dataset.train_labels.numpy()

    # sort labels
    idx_labels = np.vstack((idxs, lst_lable))
    idx_labels = idx_labels[:,idx_labels[1,:].argsort()]
    idxs = idx_labels[0,:]

    # divide and assign
    for num in range(number_users):
        random_set = set(np.random.choice(idx_shard_lst, 2, replace=False))
        idx_shard_lst = list(set(idx_shard_lst) - random_set)
        for rand in random_set:
            dict_users[num] = np.concatenate((dict_users[num], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
