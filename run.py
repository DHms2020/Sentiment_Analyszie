from my_dataset import MyDataset, PadCollate, PadCollateAllBatchSame,MyDatasetAllInRAM
import torch.nn as nn
import torch
from LSTM import LSTM
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import time
from tensorboardX import SummaryWriter
import numpy as np

# 配置数据集路径
BASDIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.abspath(os.path.join(BASDIR, ".data", "imdb", "aclImdb", "train"))
test_path = os.path.abspath(os.path.join(BASDIR, ".data", "imdb", "aclImdb", "test"))
# 配置网络超参
INPUT_DIM = 25002
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = 25001
# 配置tensorboard输出路径
Tensorboard_LSTM_path = "Bidirection_LSTM_tensboard"


# 准确率函数
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    accurate = correct.sum() / len(correct)
    return accurate


# 时间格式转换函数
def time_format(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, train_loader, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in tqdm(train_loader):
        data, labels, lengths = batch
        # print(data.shape)
        # print(lengths)
        optimizer.zero_grad()
        device = torch.device('cuda')
        data = data.to(device)
        labels = labels.to(device)
        predictions = model(data, lengths).squeeze(1)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(train_loader), epoch_acc / len(train_loader)
    # print(epoch_loss / len(train_loader), epoch_acc / len(train_loader))


def test(model, test_loader, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            data, labels, lengths = batch
            device = torch.device('cuda')
            data = data.to(device)
            labels = labels.to(device)
            predictions = model(data, lengths).squeeze(1)
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(train_loader), epoch_acc / len(train_loader)


if __name__ == "__main__":
    # model=LSTM()
    # model.load_state_dict(torch.load(model_path))
    # pretrain_emb = np.load('glove_embedding.npy')
    model = LSTM(vocab_size=INPUT_DIM, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
                 n_layers=N_LAYERS, bidirectional=BIDIRECTIONAL, dropout_rate=DROPOUT, pad_idx=PAD_IDX)
    # model.embedding.weight.data.copy_(torch.from_numpy(pretrain_emb))
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    # train_data = MyDataset(train_path)
    # test_data = MyDataset(test_path)
    train_data = MyDatasetAllInRAM("train_data.pkl")
    test_data = MyDatasetAllInRAM("test_data.pkl")
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=PadCollate(dim=0))
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=PadCollate(dim=0))
    # # 定义tensorboard
    writer = SummaryWriter(Tensorboard_LSTM_path)

    epoch = 5
    start_time = time.time()
    for i in range(epoch):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        writer.add_scalar('train_loss', train_loss, global_step=i)
        writer.add_scalar('train_acc', train_acc, global_step=i)
        print("epoch {}-->average_loss:{}  average_acc:{}".format(i, train_loss, train_acc))
    end_time = time.time()
    mini, sec = time_format(start_time, end_time)
    writer.add_scalar('total time--min', mini)
    writer.add_scalar('total time--sec', sec)
    torch.save(model.state_dict(), './models/LSTM_para.pth')
    print("5 epoch-->Total Time Used: {} min:{} sec".format(mini, sec))
    test_loss, test_acc = test(model, test_loader, criterion)
    print("TestResult-->average_loss:{}  average_acc:{}".format(test_loss, test_acc))
    writer.add_scalar('test_loss', test_loss)
    writer.add_scalar('test_acc', test_acc)
