# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

# module for dependency-embedded recursive neural network (DERNN)
class DERNN(nn.Module):
    def __init__(self, in_dim, h_dim, num_dep, w2v_model):
        super(DERNN, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.w2v_model = w2v_model
        self.iux = nn.Linear(self.in_dim, 2 * self.h_dim)
        self.iuh = nn.Linear(self.h_dim, 2 * self.h_dim)
        self.iue = nn.Linear(self.h_dim, 2 * self.h_dim)
        self.fx = nn.Linear(self.in_dim, self.h_dim)
        self.fh = nn.Linear(self.h_dim, self.h_dim)
        self.fe = nn.Linear(self.h_dim, self.h_dim)
        # dependency embedding
        self.demb = nn.Embedding(num_dep, self.h_dim)
        nn.init.xavier_uniform(self.demb.weight)

    def node_forward(self, inputs, child_h):
        child_h_sum = torch.sum(child_h[0], dim=0, keepdim=True)
        child_e = self.demb(Var(torch.LongTensor(child_h[1])))
        child_e_sum = torch.sum(child_e, dim=0, keepdim=True)

        iu = self.iux(inputs[0]) + self.iuh(child_h_sum) + self.iue(child_e_sum)
        i, u = torch.split(iu, iu.size(1)//2, dim=1)
        i, u = F.sigmoid(i), F.tanh(u)

        f = F.sigmoid(
                self.fh(child_h[0]) +
                self.fx(inputs[0]).repeat(len(child_h[1]), 1) +
                self.fe(child_e)
            )
        fxh = torch.mul(f, child_h[0])

        h = F.tanh(torch.mul(i, u) + torch.sum(fxh, dim=0, keepdim=True))
        return h, inputs[1]

    def forward(self, tree):
        inputs = (Var(torch.from_numpy(self.w2v_model[tree[0][0]]).float().view(1, -1)), tree[0][1])
        children_state = [self.forward(tree[i+1]) for i in range(len(tree)-1)]

        if len(tree) == 1:
            child_h = (Var(torch.zeros(1, self.h_dim)), (0,))
        else:
            child_h, child_dep = zip(* children_state)
            child_h = (torch.cat(child_h, dim=0), child_dep)

        child_state = self.node_forward(inputs, child_h)
        return child_state


# the whole module
class GatedDRGT(nn.Module):
    def __init__(self, args, w2v_model):
        super(GatedDRGT, self).__init__()
        self.h_dim = args.h_dim
        # DERNN
        self.dernn = DERNN(args.in_dim, args.h_dim, args.num_dep, w2v_model)
        # GRU
        self.gru = nn.GRU(args.h_dim, args.h_dim, batch_first=True)
        # Topic
        self.topic_mlp = nn.Linear(args.num_topic, args.h_dim)
        # Gate
        self.W_semgate = nn.Linear(args.h_dim, args.h_dim)
        self.U_semgate = nn.Linear(args.h_dim, args.h_dim)
        self.W_topicgate = nn.Linear(args.h_dim, args.h_dim)
        self.U_topicgate = nn.Linear(args.h_dim, args.h_dim)
        # output
        self.classifier = nn.Linear(args.h_dim, args.num_class)

    def forward(self, parser_data, topic_vec):
        # DERNN
        sent_vec = []
        for i in range(len(parser_data)):
            sent_h, dep = self.dernn(parser_data[i])
            sent_vec.append(sent_h)
        sent_vec = torch.cat(sent_vec, dim=0)
        # GRU
        doc_vec, _ = self.gru(sent_vec.unsqueeze(0))
        doc_vec = doc_vec[:, -1, :]
        # Topic
        topic_h = F.tanh(self.topic_mlp(topic_vec))
        # Gate
        gate_sem = F.sigmoid(self.W_semgate(doc_vec)+self.U_semgate(topic_h))
        gate_topic = F.sigmoid(self.W_topicgate(doc_vec)+self.U_topicgate(topic_h))
        fea_vec = F.tanh(torch.mul(doc_vec, gate_sem) + torch.mul(topic_h, gate_topic))
        # output
        out = self.classifier(fea_vec)
        out = F.log_softmax(out, dim=1)
        return out