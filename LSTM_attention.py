# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_idx, hidden_dim, n_layers, dropout_rate, num_class):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            bidirectional=True, batch_first=True, dropout=dropout_rate)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
        self.w = nn.Parameter(torch.zeros(hidden_dim * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_dim * 2, num_class)
        # self.fc = nn.Linear(hidden_dim / 2, num_class)

    def forward(self, text, text_length):
        emb = self.embedding(text)
        packed_embedded = pack_padded_sequence(emb, text_length.cpu(), batch_first=True, enforce_sorted=False)
        H, _ = self.lstm(packed_embedded)
        H, output_lengths = nn.utils.rnn.pad_packed_sequence(H)
        M = self.tanh1(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, 0)
        out = F.relu(out)
        out = self.fc1(out)
        # out = self.fc(out)
        return out
