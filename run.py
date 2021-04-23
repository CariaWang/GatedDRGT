# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pickle
import numpy as np
from model import GatedDRGT
from parameter import parse_args

args = parse_args()

# load data
w2v_model = pickle.load(open('dataset/w2v_model.pickle','rb'))
parser_data = pickle.load(open('dataset/parser_data.pickle','rb'))
label = np.load('dataset/label.npy')

doc_topics = np.load('dataset/doc_topics.npy')
topic_tensor = Variable(torch.from_numpy(doc_topics).float())

# network
net = GatedDRGT(args, w2v_model)
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.momentum, 0.999), weight_decay=args.wd)
criterion = nn.KLDivLoss()

for epoch in range(args.num_epoch):
    print('Epoch:', epoch+1)
    indices = torch.randperm(args.train_size)
    running_loss, running_acc, running_ap = 0., 0., 0.
    net.train()
    optimizer.zero_grad()
    for i in range(args.train_size):
        out = net(parser_data[indices[i]], topic_tensor[indices[i]].view(1,-1))
        y = Variable(torch.from_numpy(label[indices[i]]).view(1,-1))
        # running acc
        _, pred = torch.max(out, dim=1)
        _, truth = torch.max(y, dim=1)
        num_correct = (pred == truth).sum()
        running_acc += num_correct.data[0]
        # running ap
        out_exp = np.power(np.e, out.data.numpy())
        y_numpy = y.data.numpy()
        running_ap += np.corrcoef(out_exp[0], y_numpy[0])[0, 1]
        # running loss
        loss = criterion(out, y)
        running_loss += loss.data[0]
        loss.backward()
        if (i+1) % args.batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
    print('TrainLoss: {:.3f}, TrainAcc: {:.4f}, TrainAP: {:.4f}'.format(
                running_loss/args.train_size, running_acc/args.train_size, running_ap/args.train_size))

    # test
    net.eval()
    eval_loss, eval_acc, eval_ap = 0., 0., 0.
    for i in range(args.test_size):
        out = net(parser_data[i+args.train_size], topic_tensor[i+args.train_size])
        y = Variable(torch.from_numpy(label[i+args.train_size]).view(1,-1))
        # loss
        loss = criterion(out, y)
        eval_loss += loss.data[0]
        # acc
        _, pred = torch.max(out, 1)
        _, truth = torch.max(y, 1)
        num_correct = (pred == truth).sum()
        eval_acc += num_correct.data[0]
        # ap
        out_exp = np.power(np.e, out.data.numpy())
        y_numpy = y.data.numpy()
        eval_ap += np.corrcoef(out_exp[0], y_numpy[0])[0, 1]
    print('TestLoss: {:.3f}, TestAcc: {:.4f}, TestAP: {:.4f}'.format(
        eval_loss/args.test_size, eval_acc/args.test_size, eval_ap/args.test_size))