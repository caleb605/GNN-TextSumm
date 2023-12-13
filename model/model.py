# -*- coding: utf-8 -*-

import os, sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import GNN_Encoder
from .transformer import Encoder, Decoder
from misc import whoami

class GNN_Transformer(nn.Module):
    def __init__(self, config):
        super(__class__, self).__init__()
        vocab_size = config.lm.embeddings.word_embeddings.num_embeddings
        d_model = config.lm.embeddings.word_embeddings.embedding_dim

        self.gnn_encoder = GNN_Encoder(config) # 1st encoder: GNN Encoder
        self.tns_encoder = Encoder(config)     # 2nd encoder: Transformer Encoder
        self.tns_decoder = Decoder(config)     # deocoder   : Transformer Decoder
        self.out = nn.Linear(d_model, vocab_size)
        pass

    def forward(self, *batch):
        T, TN, B, BN, HM, S_, SM, V, A1, A2 = batch

        g_outputs = self.gnn_encoder(T, TN, B, BN, V, A1, A2)
        e_outputs = self.tns_encoder(g_outputs, HM)

        d_outputs = self.tns_decoder(S_, e_outputs, HM, SM)
        output = self.out(d_outputs)
        return output

    def get_word_embedding(self, word):
        return self.gnn_encoder.get_word_embedding(word)

    pass


def get_model(config):
    model = GNN_Transformer(config)
    return model if config.no_cuda else model.cuda()


def main():
    pass


if __name__ == '__main__':
    main()
    pass
