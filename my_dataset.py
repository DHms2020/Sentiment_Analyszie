from torch.utils.data import Dataset
import torch
import os
import spacy
import pickle as pkl

vocab_dic = pkl.load(open((os.path.abspath(os.path.join(".", "vocab.pkl"))), 'rb'))


class PadCollateAllBatchSame:
    def __init__(self, dim=0, pad_size=200):
        self.dim = dim
        self.pad_size = pad_size

    def pad_collate(self, batch):
        batch = list(map(lambda x_y_z:
                         (self.pad_or_cut(x_y_z[0], pad=self.pad_size, dim=self.dim), x_y_z[1], x_y_z[2]), batch))
        # stack all
        xs = torch.stack(list(map(lambda x: x[0][0], batch)), dim=0)
        ys = torch.tensor(list(map(lambda x: x[1], batch)))
        zs = torch.tensor(list(map(lambda x: self.pad_size if x[0][1] else x[2], batch)))
        return xs, ys, zs

    def __call__(self, batch):
        return self.pad_collate(batch)

    @staticmethod
    def pad_or_cut(vec, pad, dim):
        global vocab_dic
        is_cut_flag = False
        if len(vec) < pad:
            pad_size = pad - len(vec)
            vec = torch.tensor(vec)
            pad_vec = torch.full([pad_size], vocab_dic.get('<PAD>'))
            vec_processed = torch.cat([vec, pad_vec], dim=dim)
        else:
            is_cut_flag = True
            vec = vec[:pad]
            vec_processed = torch.tensor(vec)
        return vec_processed, is_cut_flag


class PadCollate:
    def __init__(self, dim=0):
        self.dim = dim

    def pad_collate(self, batch):
        # find longest sequence
        max_len = max(map(lambda x: len(x[0]), batch))
        # pad according to max_len
        batch = list(map(lambda x_y_z:
                         (self.pad_tensor(x_y_z[0], pad=max_len, dim=self.dim), x_y_z[1], x_y_z[2]), batch))
        # stack all
        xs = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        ys = torch.tensor(list(map(lambda x: x[1], batch)))
        zs = torch.tensor(list(map(lambda x: x[2], batch)))
        return xs, ys, zs

    def __call__(self, batch):
        return self.pad_collate(batch)

    @staticmethod
    def pad_tensor(vec, pad, dim):
        global vocab_dic
        pad_size = pad - len(vec)
        vec = torch.tensor(vec)
        pad_vec = torch.full([pad_size], vocab_dic.get('<PAD>'))
        return torch.cat([vec, pad_vec], dim=dim)


class MyDataset(Dataset):
    # 初始化并读取词表
    def __init__(self, data_path):
        self.lable_name = {"neg": 0, "pos": 1}
        self.info = self.get_path_info(data_path, self.lable_name)
        # self.vocab_dic = self.load_vocab(os.path.abspath(os.path.join(".", "vocab.pkl")))
        # 加载分词库手工添加停用词去除前100的无意义符号
        self.nlp = spacy.load('en_core_web_sm')
        self.symbol_del = {",", ".", "\"", "-", "/><br", "(", ")", "!", "\'", "?", "...", ":", "<", "br", ";", "*", "/",
                           "--", "&"}
        self.nlp.Defaults.stop_words |= self.symbol_del

    # 根据pad_size截断或者填充(pytorch1.4以上版本会报错，改为用dataloader里的collate_fn)
    def __getitem__(self, index):
        global vocab_dic
        text_path, text_label = self.info[index]
        content = []
        content_id = []
        with open(text_path, "r", encoding='UTF-8') as f:
            for line in f:
                for token in self.nlp(line):
                    if not self.nlp.vocab[token.text].is_stop:
                        content.append(token.text)
        length = len(content)
        # if self.pad_size:
        #     if length < self.pad_size:
        #         content.extend('<PAD>' * (self.pad_size - len(content)))
        #     else:
        #         content = content[:self.pad_size]
        #         length = self.pad_size
        for word in content:
            content_id.append(vocab_dic.get(word, vocab_dic.get('<UNK>')))
        return content_id, text_label, length

    def __len__(self):
        return len(self.info)

    # 遍历文件
    @staticmethod
    def get_path_info(data_path, label_name):
        path_info = []
        for root, _, files in os.walk(data_path):
            if root.endswith("neg") or root.endswith("pos"):
                for file in files:
                    file_path = os.path.join(root, file)
                    label = label_name.get(root[-3:])
                    path_info.append((file_path, float(label)))
        return path_info


class MyDatasetAllInRAM(Dataset):
    def __init__(self, pkl_name):
        print("data---loadding------\n")
        self.data = pkl.load(open((os.path.abspath(os.path.join(".", pkl_name))), 'rb'))
        print("data--has--been--loadded\n")

    def __getitem__(self, index):
        text_label = self.data[index][0]
        content_id = self.data[index][1]
        length = len(self.data[index][1])
        return content_id, text_label, length

    def __len__(self):
        return len(self.data)