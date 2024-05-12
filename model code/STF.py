import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        B, n_heads, len1, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class SMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SMultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, input_Q, input_K, input_V):
        B, N, T, C = input_Q.shape
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)
        context = ScaledDotProductAttention()(Q, K, V)
        context = context.permute(0, 3, 2, 1, 4)
        context = context.reshape(B, N, T, self.heads * self.head_dim)
        output = self.fc_out(context)
        return output

class LSTMcell(nn.Module):
    def __init__(self, input_size, hidden_layer_size):
        super(LSTMcell, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_layer_size,
            batch_first=True,
        )
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_layer_size),
                torch.zeros(1, batch_size, self.hidden_layer_size))
    def forward(self, input_seq):
        self.hidden = self.init_hidden(input_seq.shape[0])
        lstm_out, self.hidden = self.lstm(input_seq, self.hidden)
        return lstm_out

class Transformer_cell(nn.Module):
    def __init__(self, embedsize, nodes, heads, dropout=0):
        super(Transformer_cell, self).__init__()
        self.nodes = nodes
        self.SAT = SMultiHeadAttention(embedsize, heads)
        self.relu = nn.ReLU()
        self.feed_forward = nn.Sequential(
            nn.Linear(embedsize, 2 * embedsize),
            nn.ReLU(),
            nn.Linear(2 * embedsize, embedsize),
        )
        self.norm1 = nn.LayerNorm(embedsize)
        self.norm2 = nn.LayerNorm(embedsize)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res_S = x
        x_S_Q = x
        x_S_K = x
        x_S_V = x

        x_S = self.SAT(x_S_Q, x_S_K, x_S_V)
        x_S = self.dropout(self.norm1(x_S + res_S))
        x_feed = self.feed_forward(x_S)
        x_S = self.dropout(self.norm2(x_feed + x_S))

        return x_S

class STF(nn.Module):
    def __init__(self, embedsize, nodes, heads, pred_len, time_step, dropout=0):
        super(STF, self).__init__()
        self.nodes = nodes
        self.Transformer = Transformer_cell(embedsize, nodes, heads, dropout)
        self.lstm = LSTMcell(nodes, embedsize)
        self.conv1 = nn.Conv2d(1, embedsize, 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, nodes, time_step, embedsize), requires_grad=True)
        self.final_c = nn.Sequential(
            nn.Linear(embedsize, embedsize*2),
            nn.ReLU(),
            nn.Linear(embedsize*2, nodes)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(embedsize*nodes, embedsize*2),
            nn.ReLU(),
            nn.Linear(embedsize*2, nodes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1)
        input = self.conv1(x)
        input = input.permute(0, 2, 3, 1)
        input = input + self.pos_embed
        B, N, T, H = input.shape
        x_S = self.ST1(input)
        x_S = x_S.permute(0, 2, 1, 3)
        x_S = x_S.reshape(B, T, -1)
        x_S = self.fc1(x_S)
        out = self.lstm(x_S)
        out = out[:, -1, :].reshape(B, -1)
        out = self.final_c(out)
        return out
