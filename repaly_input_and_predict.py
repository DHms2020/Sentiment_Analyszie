import torch
import os
import LSTM
import LSTM_attention
import Transformer
import Bert_GRU
import spacy
import pickle as pkl
from transformers import BertTokenizer, BertModel

# 手工添加停用词-特殊符号
symbol_del = {",", ".", "\"", "-", "/><br", "(", ")", "!", "\'", "?", "...", ":", "<", "br", ";", "*", "/", "--", "&"}
vocab = pkl.load(open((os.path.abspath(os.path.join(".", "vocab.pkl"))), 'rb'))
nlp = spacy.load('en_core_web_sm')
nlp.Defaults.stop_words |= symbol_del


def case_LSTM():
    model = LSTM.LSTM(vocab_size=25002, embedding_dim=300, hidden_dim=256, output_dim=1,
                 n_layers=2, bidirectional=True, dropout_rate=0.5, pad_idx=25001)
    pretained_model = torch.load("./models/LSTM_para.pth", map_location='cpu')
    model.load_state_dict(pretained_model)
    model.eval()

    while 1:
        review = input("please input your comment; or exit with 'quit':")
        if (review == "quit"):
            break
        else:
            content_id = []
            for token in nlp(review):
                if not nlp.vocab[token.text].is_stop:
                    content_id.append(vocab.get(token.text, vocab.get('<UNK>')))
            length = len(content_id)

            if (length == 0):
                print("pleas add more information\n")
            else:
                content_id_tensor = torch.LongTensor(content_id)
                content_id_tensor = content_id_tensor.unsqueeze(0)  # Transformer演示时注释掉这行
                length_tensor = torch.LongTensor([length])
                predictions = torch.sigmoid(model(content_id_tensor, length_tensor))
                print(predictions.item())


def case_LSTM_attention():
    model = LSTM_attention.Model(vocab_size=25002, embedding_dim=300, hidden_dim=256, num_class=1,
                  n_layers=2, dropout_rate=0.5, pad_idx=25001)
    pretained_model = torch.load("./models/LSTM_attention_para.pth", map_location='cpu')
    model.load_state_dict(pretained_model)
    model.eval()

    while 1:
        review = input("please input your comment; or exit with 'quit':")
        if (review == "quit"):
            break
        else:
            content_id = []
            for token in nlp(review):
                if not nlp.vocab[token.text].is_stop:
                    content_id.append(vocab.get(token.text, vocab.get('<UNK>')))
            length = len(content_id)

            if (length == 0):
                print("pleas add more information\n")
            else:
                content_id_tensor = torch.LongTensor(content_id)
                content_id_tensor = content_id_tensor.unsqueeze(0)  # Transformer演示时注释掉这行
                length_tensor = torch.LongTensor([length])
                predictions = torch.sigmoid(model(content_id_tensor, length_tensor))
                print(predictions.item())


def case_Transformer():
    model = Transformer.Model(vocab_len=25002, padding_id=25001, pad_size=200, dropout_rate=0.5, num_class=1,
                  device=torch.device('cpu'))
    pretained_model = torch.load("./models/Transformer_para.pth", map_location='cpu')
    model.load_state_dict(pretained_model)
    model.eval()

    while 1:
        review = input("please input your comment; or exit with 'quit':")
        if (review == "quit"):
            break
        else:
            content_id = []
            for token in nlp(review):
                if not nlp.vocab[token.text].is_stop:
                    content_id.append(vocab.get(token.text, vocab.get('<UNK>')))
            length = len(content_id)

            if (length == 0):
                print("pleas add more information\n")
            else:
                content_id_tensor = torch.LongTensor(content_id)
                # Transformer的输入需要填充到PAD_SIEZ，这里是默认200
                pad_size = 200 - length
                pad_vec = torch.full([pad_size], 25001)
                content_id_tensor = torch.cat([content_id_tensor, pad_vec], dim=0)
                content_id_tensor = content_id_tensor.unsqueeze(0)
                predictions = torch.sigmoid(model(content_id_tensor))
                print(predictions.item())


def case_Bert_GRU():
    tokenizer = BertTokenizer.from_pretrained("./Bert_pretrained/bert-base-uncased")
    bert = BertModel.from_pretrained('./Bert_pretrained/bert-base-uncased')
    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']  #Bert仅支持512的最大长度

    model = Bert_GRU.BERTGRUSentiment(bert=bert, hidden_dim=256, output_dim=1, n_layers=2, bidirectional=True, dropout=0.25)
    pretained_model = torch.load("./models/BERT+GRU.pth", map_location='cpu')
    model.load_state_dict(pretained_model)
    model.eval()

    while 1:
        review = input("please input your comment; or exit with 'quit':")
        if (review == "quit"):
            break
        else:
            # Bert的分词需要用模型内置的分词器，不能用spacy
            tokens = tokenizer.tokenize(review)
            length=len(tokens)
            if (length == 0):
                print("pleas add more information\n")
            else:
                tokens = tokens[:max_input_length - 2]  # 开头和结尾还要加特殊符号
                init_token_idx = tokenizer.cls_token_id
                eos_token_idx = tokenizer.sep_token_id
                indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]

                content_id_tensor = torch.LongTensor(indexed)
                content_id_tensor = content_id_tensor.unsqueeze(0)
                predictions = torch.sigmoid(model(content_id_tensor))
                print(predictions.item())


def default():
    print("Wrong model chose,please try again\n")


if __name__ == "__main__":
    # model = LSTM(vocab_size=25002, embedding_dim=300, hidden_dim=256, output_dim=1,
    #              n_layers=2, bidirectional=True, dropout_rate=0.5, pad_idx=25001)        #BiLSTM
    # model = Model(vocab_size=25002, embedding_dim=300, hidden_dim=256, num_class=1,
    #               n_layers=2, dropout_rate=0.5, pad_idx=25001)          #BiLSTM+attention
    # model = Model(vocab_len=25002, padding_id=25001, pad_size=200, dropout_rate=0.5,num_class=1,device=torch.device('cpu'))   #Transformer
    # model = BERTGRUSentiment(bert=bert, hidden_dim=256, output_dim=1, n_layers=2, bidirectional=True, dropout=0.25)
    # pretained_model = torch.load("./models/LSTM_para.pth", map_location='cpu')
    # pretained_model = torch.load("./models/LSTM_attention_para.pth", map_location='cpu')
    # pretained_model = torch.load("./models/Transformer_para.pth", map_location='cpu')
    # pretained_model = torch.load("./models/BERT+GRU.pth", map_location='cpu')
    # model.load_state_dict(pretained_model)
    # model.eval()
    # print(next(model.parameters()).device)
    models={'1': case_LSTM,
            '2': case_LSTM_attention,
            '3': case_Transformer,
            '4': case_Bert_GRU}
    while 1:
        choice=input("please input the model name you chose\n1 : LSTM\n2 : LSTM_attention\n3 : Transformer\n4 : Bert+GRU\n")
        models.get(choice,default)()


