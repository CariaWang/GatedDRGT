# -*- coding: utf-8 -*-
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='GatedDRGT for Social Emotion Classification')

    # dataset
    parser.add_argument('--data_size', default=5257, type=int, help='Dataset size')
    parser.add_argument('--train_size', default=3109, type=int, help='Train set size')
    parser.add_argument('--test_size', default=2148, type=int, help='Test set size')
    parser.add_argument('--num_class', default=6, type=int, help='Number of classes')

    # model arguments
    parser.add_argument('--in_dim', default=100, type=int, help='Size of input word vector')
    parser.add_argument('--h_dim', default=200, type=int, help='Size of TreeLSTM cell state')
    parser.add_argument('--num_topic', default=30, type=int, help='Number of topics')
    parser.add_argument('--num_dep', default=15, type=int, help='Number of dependency relations')

    # training arguments
    parser.add_argument('--num_epoch', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=20, type=int, help='batchsize for optimizer updates')
    parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', default=5e-5,  type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.99,  type=float)

    args = parser.parse_args()
    return args
