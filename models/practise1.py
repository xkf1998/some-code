import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import models
import matplotlib.pyplot as plt

torch.manual_seed(1)  # 没懂

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255.
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # number of filter
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 在2*2的空间内进行采样
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class rnn(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=28,  # 每行的像素点
            hidden_size= 64,
            num_layer= 1,
            batch_first=True,
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, b_y)in enumerate(train_loader):
        b_x = x.view(-1, 28, 28)
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class luong_attention(nn.Module):

    def __init__(self, hidden_size, emb_size, pool_size = 0):
        super(luong_attention, self).__init__()
        self.hidden_size, self.emb_size, self.pool_size = hidden_size, emb_size, pool_size
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        if pool_size > 0:
            self.linear_out = maxout(2*hidden_size + emb_size, hidden_size, pool_size)
        else:
            self.linear_out = nn.Sequential(nn.Linear(2*hidden_size + emb_size, hidden_size),nn.Tanh())
        self.softmax = nn.Softmax(dim=1)

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, h, x):
        gamma_h = self.linear_in(h).unsqueeze(2)
        weights = torch.bmm(self.context, gamma_h).squeeze(2)
        weights = self.softmax(weights)
        c_t = torch.bmm(weights.unsqueeze(1), self.context).squeeze(1)
        output = self.linear_out(torch.cat([c_t, h, x], 1))

        return output, weights

class bahdanau_attention(nn.Module):

    def __init__(self, hidden_size, emb_size, pool_size = 0):
        super(bahdanau_attention, self).__init__()
        self.linear_encoder = nn.Linear(hidden_size, hidden_size)
        self.linear_decoder = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, 1)
        self.linear_r = nn.Linear(hidden_size * 2 + emb_size, hidden_size * 2)
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.softmax = nn.Softmax(dim = 1)
        self.tanh = nn.Tanh()

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, h, x):
        gamma_encoder = self.linear_encoder(self.context)
        gamma_decoder = self.linear_decoder(h).unsqueeze(1)
        weights = self.linear_v(self.tanh(gamma_encoder + gamma_decoder)).squeeze(2)
        weights = self.softmax(weights)
        c_t = torch.bmm(weights.unsqueeze(1), self.context).squeeze(1)
        r_t = self.linear_r(torch.cat([c_t, h, x], dim = 1))
        output = r_t.view(-1, self.hidden_size, 2).max(2)[0]
        return output, weights

class maxout(nn.Module):
    def __init__(self, in_feature, out_feature, pool_size):
        super(maxout, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.pool_size = pool_size
        self.linear = nn.Linear(in_feature, out_feature * pool_size)

    def forward(self, x):
        output = self.linear(x)
        output = output.view(-1, self.out_feature, self.pool_size)
        output = output.max(2)[0]
        return output


class rnn_encoder(nn.Module):

    def __init__(self, config, embedding = None):
        super(rnn_encoder, self).__init__()

        self.embedding = embedding if embedding is not None else nn.Embedding(config.src_vocab_size, config.emb_size)
        self.hidden_size = config.hidden_size
        self.config = config

        if config.swish:
            self.sw1 = nn.Sequential(nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1, padding=0), nn.BatchNorm1d(config.hidden_size), nn.ReLU())
            self.sw3 = nn.Sequential(nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm1d(config.hidden_size),
                                     nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=0), nn.ReLU(), nn.BatchNorm1d(config.hidden_size))
            self.sw33 = nn.Sequential(nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm1d(config.hidden_size),
                                      nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=0), nn.ReLU(), nn.BatchNorm1d(config.hidden_size),
                                      nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=0), nn.ReLU(), nn.BatchNorm1d(config.hidden_size))
            self.linear = nn.Sequential(nn.Linear(2*config.hidden_size, 2*config.hidden_size), nn.GLU(), nn.Dropout(config.dropout))
            self.filter_linear = nn.Linear(3 * config.hidden_size, config.hidden_size)
            self.tahn = nn.Tanh()
            self.sigmoid = nn.Sigmoid()

        if config.selfatt:
            if config.attention == 'None':
                self.attention = None
            elif config.attention == 'bahdanau':
                self.attention = models.bahdanau_attention(config.hidden_size, config.emb_size, config.pool_size)
            elif config.attention == 'luong':
                self.attention = models.luong_attention(config.hidden_size, config.emb_size, config.pool_size)
            elif config.attention == 'luong_gate':
                self.attention = models.luong_gate_attention(config.hidden_size, config.emb_size)

        if config.cell == 'gru':
            self.rnn = nn.GRU(input_size=config.emb_size, hidden_size=config.hidden_size,
                              num_layers=config.enc_num_layers, dropout=config.dropout,
                              bidirectional=config.bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                               num_layers=config.enc_num_layers, dropout=config.dropout,
                               bidirectional=config.bidirectional)


    def forward(self, inputs, lengths):
        embs = pack(self.embedding(inputs),lengths)
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        if self.config.bidirectional:
            if self.config.swish:
                outputs = self.linear(outputs)
            else:
                outputs = outputs[:, :, :self.config.hidden_size] + outputs[:, :, self.config.hidden_size:]

        if self.config.swish:
            outputs = outputs.transpose(0, 1).transpose(1, 2)
            conv1 = self.sw1(outputs)
            conv3 = self.sw3(outputs)
            conv33 = self.sw33(outputs)
            conv = torch.cat((conv1, conv3, conv33), 1)
            conv = self.filter_linear(conv.transpose(1, 2))
            if self.config.selfatt:
                conv = conv.transpose(0, 1)
                outputs = outputs.transpose(1, 2).transpose(0, 1)
            else:
                gate = self.sigmoid(conv)
                outputs = outputs * gate.transpose(1, 2)
                outputs = outputs.transpose(1, 2).transpose(0, 1)

        if self.config.selfatt:
            self.attention.init_context(context=conv)
            out_attn, weights = self.attention(conv, selfatt=True)
            gate = self.sigmoid(out_attn)
            outputs = outputs * gate

        if self.config.cell == 'gru'
            state = state[:self.config.dec_num_layers]
        else:
            state = (state[0][::2], state[1][::2])

        return outputs, state

class rnn_decoder(nn.Module):

    def __init__(self, config, embedding=None, use_attention=True):
        super(rnn_decoder,self).__init__()
        self.embedding = embedding if embedding is not None else nn.Embedding(config.tgt_vocab_size, config.embsize)

        input_size = config.emb_size

        if config.cell =='gru':
            self.rnn = StackedGRU(input_size=input_size, hidden_size=config.hidden_size,
                                  num_layers=config.dec_num_layers, dropout=config.dropout)
        else:
            self.rnn = StackedLSTM(input_size= input_size, hidden_size=config.hidden_size,
                                   num_layers=config.dec_num_layers, dropout=config.dropout)

        self.linear = nn.Linear(config.hidden_size, config.tgt_vocab_size)
        self.linear_ = nn.Linear(config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

        if not use_attention or config.attention == 'None':
            self.attention = None
        elif config.attention == 'bahdanau':
            self.attention = models.bahdanau_attention(config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong':
            self.attention = models.luong_attention(config.hidden_size,config.emb_size,config.pool_size)
        elif config.attention == 'luong_gate':
            self.attention = models.luong_gate_attention(config.hidden_size, config.emb_size, prob=config.config.dropout)

        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, input, state):
        embs = self.embedding(input)
        output, state = self.rnn(embs, state)
        if self.attention is not None:
            if self.config.attention == 'luong_gate':
                output, attn_weights = self.attention(output)
            else:
                output, attn_weights = self.attention(output, embs)
        else:
            attn_weights = None

        output = self.compute_score(output)

        return output, state, attn_weights

    def compute_score(self, hiddens):
        scores = self.linear(hiddens)
        return scores

class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hiddensize, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(inputsize, hiddensize))
            input_size = hiddensize

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)




