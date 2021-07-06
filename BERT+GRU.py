import torch
import random
import numpy as np
from transformers import BertTokenizer,BertModel
from torchtext.legacy import data
from torchtext.legacy import datasets
import torch.nn as nn
import torch.optim as optim
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm

SEED = 1234
BATCH_SIZE = 64

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# 配置tensorboard输出路径
Tensorboard_LSTM_path = "BERT+GRU_tensboard"


class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text = [batch size, sent len]

        with torch.no_grad():
            embedded = self.bert(text)[0]

        # embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        output = self.out(hidden)

        # output = [batch size, out dim]

        return output


#根据bert加载分词函数
def tokenize_and_cut(sentence):
    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens

#准确度函数
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def time_format(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in tqdm(iterator):
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def test(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(iterator):
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

if __name__ == "__main__":
    #初始化一些超参数
    device = torch.device('cuda')
    bert = BertModel.from_pretrained('bert-base-uncased')
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25
    #取得特殊词的ID值
    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id
    #加载Field
    TEXT = data.Field(batch_first=True,
                      use_vocab=False,
                      tokenize=tokenize_and_cut,
                      preprocessing=tokenizer.convert_tokens_to_ids,
                      init_token=init_token_idx,
                      eos_token=eos_token_idx,
                      pad_token=pad_token_idx,
                      unk_token=unk_token_idx)

    LABEL = data.LabelField(dtype=torch.float)
    #建立datasets
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    LABEL.build_vocab(train_data)
    #建立数据迭代器
    train_iterator,  test_iterator = data.BucketIterator.splits(
        (train_data, test_data),
        batch_size=BATCH_SIZE,
        device=device)

    model = BERTGRUSentiment(bert,
                             HIDDEN_DIM,
                             OUTPUT_DIM,
                             N_LAYERS,
                             BIDIRECTIONAL,
                             DROPOUT)

    for name, param in model.named_parameters():
        if name.startswith('bert'):
            param.requires_grad = False

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    # # 定义tensorboard
    writer = SummaryWriter(Tensorboard_LSTM_path)

    epoch=5
    start_time = time.time()
    for i in range(epoch):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        writer.add_scalar('train_loss', train_loss, global_step=i)
        writer.add_scalar('train_acc', train_acc, global_step=i)
        print("epoch {}-->average_loss:{}  average_acc:{}".format(i, train_loss, train_acc))
    end_time = time.time()
    mini, sec = time_format(start_time, end_time)
    writer.add_scalar('total time--min', mini)
    writer.add_scalar('total time--sec', sec)
    torch.save(model.state_dict(), './models/BERT+GRU.pth')
    print("5 epoch-->Total Time Used: {} min:{} sec".format(mini, sec))
    test_loss, test_acc = test(model, test_iterator, criterion)
    print("TestResult-->average_loss:{}  average_acc:{}".format(test_loss, test_acc))
    writer.add_scalar('test_loss', test_loss)
    writer.add_scalar('test_acc', test_acc)