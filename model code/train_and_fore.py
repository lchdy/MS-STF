import pytorch_lightning as pl
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from STF import STF
import warnings
from pytorch_lightning.callbacks import ModelCheckpoint

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

period = 31

def split_train_data(data, seq_len, pred_len):
    recent_x, recent_y, weekly_x, monthly_x \
        = list(), list(), list(), list()
    for i in range((seq_len+period-1), len(data)-pred_len+1):
        recent_x.append(np.array(data[i-seq_len:i, :]))
        recent_y.append(np.array(data[i:i+pred_len, :]))
        weekly_x.append(np.array(data[(i-7-seq_len+1):(i-7+1), :]))
        monthly_x.append(np.array(data[(i-31-seq_len+1):(i-31+1), :]))
    return np.array(recent_x), np.array(recent_y), np.array(weekly_x), np.array(monthly_x)

def split_test_data(data, seq_len, pred_len, num_test):
    recent_x, recent_y, weekly_x, monthly_x \
        = list(), list(), list(), list()
    for k in range((seq_len+period), (seq_len+period+num_test), pred_len):
        recent_x.append(np.array(data[k - seq_len:k, :]))
        recent_y.append(np.array(data[k:k + pred_len, :]))
        weekly_x.append(np.array(data[(k - 7 - seq_len + 1):(k - 7 + 1), :]))
        monthly_x.append(np.array(data[(k - 31 - seq_len + 1):(k - 31 + 1), :]))
    return np.array(recent_x), np.array(recent_y), np.array(weekly_x), np.array(monthly_x)

def accuracy(prediction, true):
    a = abs(prediction - true)
    b = a / true
    c = (prediction - true) ** 2
    return np.mean(b), np.mean(a), np.sqrt(np.mean(c))

class MS_STF(pl.LightningModule):
    def __init__(self, embed_size, num_nodes, num_heads, pred_len, time_step, learning_rate, weight_decay, dropout):
        super(MS_STF, self).__init__()
        self.embed_size = embed_size
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.pred_len = pred_len
        self.time_step = time_step
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.r_x_model = STF(self.embed_size, self.num_nodes, self.num_heads, self.pred_len, \
                             self.time_step, self.dropout)
        self.w_x_model = STF(self.embed_size, self.num_nodes, self.num_heads, self.pred_len, \
                             self.time_step, self.dropout)
        self.m_x_model = STF(self.embed_size, self.num_nodes, self.num_heads, self.pred_len, \
                                self.time_step, self.dropout)
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w3 = nn.Parameter(torch.ones(1), requires_grad=True)

        self.fc = nn.Sequential(
            nn.Linear(num_nodes, embed_size*2),
            nn.ReLU(),
            nn.Linear(embed_size*2, num_nodes*pred_len)
        )

    def forward(self, x):
        r_x = x[:, :self.time_step, :]
        w_x = x[:, self.time_step:self.time_step*2, :]
        m_x = x[:, self.time_step*2:, :]
        r_x_pred = self.r_x_model(r_x)
        w_x_pred = self.w_x_model(w_x)
        m_x_pred = self.w_x_model(m_x)
        pred = self.w1*r_x_pred + self.w2*w_x_pred + self.w3*m_x_pred
        pred = self.fc(pred)
        return pred

    def loss_func(self, true, pred):
        return torch.mean(torch.pow((true - pred), 2))

    def training_step(self, batch, batch_id):
        x, y = batch
        b = y.shape[0]
        y = y.reshape(b, -1)
        pred = self(x)
        loss = self.loss_func(y, pred)
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

class GenerateDataset(pl.LightningDataModule):
    def __init__(self, seq_len: int, pred_len: int, batch_size: int, train):
        super(GenerateDataset, self).__init__()
        self.train_data = train
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size

    def train_dataloader(self):
        r_x, r_y, w_x, m_x = split_train_data(self.train_data, self.seq_len, self.pred_len)
        r_x = torch.FloatTensor(r_x)
        w_x = torch.FloatTensor(w_x)
        m_x = torch.FloatTensor(m_x)
        train_x = torch.cat((r_x, w_x, m_x), dim=1)
        train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(r_y))
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size)

def main():
    # use the best parameter combination
    embed_size = 1
    num_node = 1
    num_heads = 1
    time_step = 1
    num_test = 1
    num_train = 1
    pred_len = 1
    epochs = 1
    batch_size = 1
    dropout = 1
    learning_rate = 1
    weight_decay = 1

    demand = pd.read_csv('data.csv', header=None)
    d = np.array(demand)
    scaler = MinMaxScaler()
    train_ = d[:num_train]
    test_ = d[-(num_test + period + time_step):, :]
    train = scaler.fit_transform(train_)
    test = scaler.transform(test_)

    data = GenerateDataset(seq_len=time_step, pred_len=pred_len, batch_size=batch_size, train=train)
    model = MS_STF(embed_size, num_node, num_heads, pred_len, time_step, learning_rate, weight_decay, dropout)

    checkpoint_callback = ModelCheckpoint(
        monitor='loss',
        filename='save-{epoch:02d}-{loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True,
        save_weights_only=True
    )
    trainer = pl.Trainer(max_epochs=epochs, enable_model_summary=False, callbacks=[checkpoint_callback])
    trainer.fit(model, data)

    test_r_x, test_r_y, test_w_x, test_m_x = split_test_data(test, seq_len=time_step, pred_len=pred_len, num_test=num_test)
    test_r_x = torch.FloatTensor(test_r_x)
    test_w_x = torch.FloatTensor(test_w_x)
    test_m_x = torch.FloatTensor(test_m_x)
    test_x_ = torch.cat((test_r_x, test_w_x, test_m_x), dim=1)
    model.eval()
    pred = model(test_x_)
    pred = pred.reshape(-1, num_node)
    print(pred)

if __name__ == '__main__':
    main()
