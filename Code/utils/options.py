import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--num_users', type=int, default=100, help="Number user have: K")
    parser.add_argument('--frac', type=float, default=0.1, help="Fraction of client for every epoch: C")
    parser.add_argument('--epochs', type=int, default=10, help="Number round training")
    parser.add_argument('--local_bs', type=int, default=10, help="Batch size at local: B")
    parser.add_argument('--local_ep', type=int, default=5, help="Number of local epochs: E")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--bs', type=int, default=128, help="Batch size test")
    parser.add_argument('--split', type=str, default='user', help="Train-test split type")

    # model arguments
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for the convolution')
    parser.add_argument('--model', type=str, default='cnn', help='Name of model')
    parser.add_argument('--kernel_num', type=int, default=9, help='Number each kind of kernel')
    parser.add_argument('--num_filters', type=int, default=32, help="Number filter")
    parser.add_argument('--norm', type=str, default='batch_norm', help="Batch_norm || Layer_norm|| None")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--num_classes', type=int, default=10, help="Number Class")
    parser.add_argument('--dataset', type=str, default='mnist', help="Name Dataset")
    parser.add_argument('--iid', action='store_true',default=True, help='Data is iid || Non-iid')
    parser.add_argument('--verbose', action='store_true', help='verbose option')
    parser.add_argument('--num_channels', type=int, default=1, help="Number channels")
    parser.add_argument('--gpu', type=int, default=0, help="0: GPU ID, -1: for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--all_clients', action='store_true', help='aggregation for all client')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
