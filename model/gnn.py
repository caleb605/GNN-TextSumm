# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from misc import whoami


class GAT_gate(nn.Module):
    def __init__(self, n_features_in, n_features_out):
        super(__class__, self).__init__()

        # default values of n_features_in:160 / n_features_out:160
        self.W = nn.Linear(n_features_in, n_features_out)
        self.A = nn.Linear(n_features_out, n_features_out, bias=False)
        self.gate = nn.Linear(n_features_out * 2, 1)

        #
        # there's need to use LeakyReLU instead of F.relu in forward() function
        #
        #self.leakyrelu = torch.nn.LeakyReLU(0.2)
        pass

    def forward(self, x, adj):
        #
        # x.size(): (batch_size, n_elms, n_features_in )
        # h.size(): (batch_size, n_elms, n_features_out)
        #
        h = self.W(x)

        #e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h,self.A), h))
        e = torch.bmm(self.A(h), h.permute(0, 2, 1))

        # e.size(): (batch_size, n_features_out, n_features_out)

        e = e + e.permute((0,2,1))
        zero_vec = -9e15 * torch.ones_like(e)

        # adj.size()       : (m, m) where m = n_features_out
        # e.size()         : (batch, n_features_out, n_features_out)
        # attention.size() : (batch, n_features_out, n_features_out)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = attention * adj

        h_prime = F.relu(torch.bmm(attention, h))

        # shape after self.gate() : (batch_size, n_features_out, 1)
        # coeff.size() = x.size() (batch_size, N, n_features_in) : N = h.size(1) = x.size(1)
        coeff = torch.sigmoid(self.gate(torch.cat([x, h_prime], -1))).repeat(1, 1, x.size(-1))

        out = coeff * x + (1 - coeff) * h_prime
        self.print_forward_info = True

        return out
    pass


class GNN_Encoder(nn.Module):
    def __init__(self, config):
        super(__class__, self).__init__()

        self.config = config
        self.device = config.device
        self.tokenizer = config.tokenizer

        self.word_embeddings = config.lm.embeddings.word_embeddings # <class 'torch.nn.modules.sparse.Embedding'>
        self.word_embedding_dim = self.word_embeddings.embedding_dim

        n_graph_layer = config.n_graph_layer  # number of Graph-Convolution (number of GNN layers: default: 4)
        d_graph_layer = config.d_graph_layer  # size of each GNN layer

        n_FC_layer = config.n_FC_layer
        d_FC_layer = config.d_FC_layer

        self.dropout_rate = config.dropout_rate

        graph_depths = [d_graph_layer for i in range(n_graph_layer)]

        self.gconv = nn.ModuleList([GAT_gate(graph_depths[i], graph_depths[i + 1]) for i in range(len(graph_depths) - 1)])

        fc_layers = list()
        for i in range(n_FC_layer):
            if i == 0:
                fc_layers.append(nn.Linear(graph_depths[-1], d_FC_layer))
                pass
            elif i == n_FC_layer - 1:
                fc_layers.append(nn.Linear(d_FC_layer, self.word_embedding_dim))
                pass
            else:
                fc_layers.append(nn.Linear(d_FC_layer, d_FC_layer))
                pass

        self.FC = nn.ModuleList(fc_layers)

        self.mu  = nn.Parameter(torch.Tensor([config.initial_mu]).float())
        self.dev = nn.Parameter(torch.Tensor([config.initial_dev]).float())

        self.emb_graph = nn.Linear(self.word_embedding_dim * 2, d_graph_layer, bias = False)
        pass

    def embed_graph(self, H, A1, A2, V):
        H = H.to(self.device)
        c_hs = self.emb_graph(H)
        A2_emb = torch.exp(-torch.pow(A2 - self.mu.expand_as(A2), 2) / self.dev) + A1

        for i in range(len(self.gconv)):
            c_hs1 = self.gconv[i](c_hs, A2_emb)
            c_hs2 = self.gconv[i](c_hs, A1)
            c_hs = c_hs2 - c_hs1
            c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
            pass

        c_hs = c_hs * V.unsqueeze(-1).repeat(1, 1, c_hs.size(-1))
        return c_hs

    def fc(self, c_hs):
        for i in range(len(self.FC)):
            c_hs = self.FC[i](c_hs)
            if i < len(self.FC) - 1:
                c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
                #c_hs = F.relu(c_hs)
                pass
            pass

        return c_hs


    def forward(self, *data):
        T, TN, B, BN, V, A1, A2 = data

        T = self.word_embeddings(T)
        B = self.word_embeddings(B)

        batch_size  = T.size(0)
        n_max_ids   = T.size(1)
        n_max_sents = B.size(1)

        n_features = self.word_embedding_dim

        H = torch.zeros(batch_size, n_max_ids + n_max_sents, n_features * 2)

        for i in range(batch_size):
            n_ids   = TN[i]
            n_sents = BN[i]

            H[i, :n_ids             , :n_features] = T[i, :n_ids  ,:]
            H[i, n_ids:n_ids+n_sents, n_features:] = B[i, :n_sents,:]
            pass

        c_hs = self.embed_graph(H, A1, A2, V)

        # embed a graph to a vector
        c_hs = self.fc(c_hs)

        return c_hs


    def get_word_embedding(self, word):
        tokens = self.tokenizer.tokenize(word)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        id_tensors = torch.tensor(ids, dtype=torch.int)

        if not self.config.no_cuda:
            id_tensors = id_tensors.to(self.device)
            pass

        embedded = self.word_embeddings(id_tensors)
        return ids, embedded

    pass

