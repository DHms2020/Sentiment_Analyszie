import os
import pickle as pkl
import spacy
import os
from tqdm import tqdm
import pickle as pkl
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch
import pickle as pkl

#
# path = os.path.dirname(os.path.abspath(__file__))
# path = os.path.join(path, ".data", "imdb", "aclImdb", "test", "neg")
#
# # print(path)
# for root, dir, files in os.walk(path):
#     print("under the {} number of files is {}".format(root, len(files)))
#     if root == os.path.join(path, "neg"):
#         print(root[-3:])
#     print(files[0])

# vocab_path = "E:\\PycharmProjects\\pytorch_practice\\vocab.pkl"
# #
# vocab = pkl.load(open(vocab_path, 'rb'))
# print(vocab.get('<PAD>'))
#
# for i, j in sorted(vocab.items(), key=lambda x: x[1], reverse=False)[:100]:
#     print(i,j)

# vocab_dict_freq = {}

# symbol_del = {",", ".", "\"", "-", "/><br", "(", ")", "!", "\'", "?", "...", ":", "<", "br", ";", "*", "/", "--", "&"}
#
# nlp = spacy.load('en_core_web_sm')
# nlp.Defaults.stop_words |= symbol_del
# count = 0
# for root, dir, files in os.walk(path):
#     for file in files:
#         with open(os.path.join(root, file), "r", encoding='UTF-8') as f:
#             for line in f:
#                 for word in nlp(line):
#                     if not nlp.vocab[word.text].is_stop:
#                         count += 1
#         break
#     break
# # print(count)
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data = [torch.tensor([9, 11, 11, 11]),
#         torch.tensor([1, 2, 3, 4]),
#         torch.tensor([5, 6, 11, 11])]
# # print(data[0])
# # seq_len = [s.size(0) for s in data]
# seq_len = [1, 4, 2]
# # data = pad_sequence(data, batch_first=True)
# data = torch.stack(data, dim=0)
# # data.to(device)
# print(data.device)
# data = pack_padded_sequence(data, seq_len, batch_first=True, enforce_sorted=False)
# print(data)

# print(spacy.__version__)
#
# train_path = os.path.abspath(
#     os.path.join(BASDIR, "..", "..", "ms635", "pytorch_practice", ".data", "imdb", "aclImdb", "train"))

# BASDIR = os.path.dirname(os.path.abspath(__file__))
# train_path = os.path.abspath(os.path.join(BASDIR, ".data", "imdb", "aclImdb"))
# print(train_path)
# max_length = 0
# max_path = ""
# min_length = 0
# min_path = ""
# for root, _, files in os.walk(train_path):
#     if root.endswith("neg") or root.endswith("pos"):
#         for file in tqdm(files):
#             with open(os.path.join(root, file), 'r', encoding='UTF-8') as f:
#                 content = []
#                 for line in f:
#                     for word in line:
#                         content.append(word)
#                     # for word in nlp(line):
#                     #    if not nlp.vocab[word.text].is_stop:
#                     #        content.append(word.text)
#             length = len(content)
#             if length > max_length:
#                 max_length = length
#                 max_path = os.path.abspath(os.path.join(root, file))
#             if length < min_length:
#                 min_length = length
#                 min_path = os.path.abspath(os.path.join(root, file))
#
# print("the min token length is {}from {}; the max token length is{} from {}".format(min_length, min_path, max_length,
#                                                                                     max_path))

# path="./models/tes.txt"
# with open(path,"w",encoding='UTF-8') as w:
#     w.write("向上吧少年")

# import numpy as np
# import matplotlib.pyplot as plt
#
# t = np.array([2, 4,6,8,10])
# # print(t)
# plt.plot(t, t, 'r')
# plt.plot(t, t ** 2, 'b')
# label = ['t', 't**2']
# plt.legend(label, loc='upper left')
# plt.show()

# def process(number):
#     is_cut_flag = False
#     if number < 10:
#         return number, is_cut_flag
#     else:
#         is_cut_flag = True
#         return number, is_cut_flag
#
#
# a = [[8, 5, 9],
#      [12, 4, 8],
#      [11, 7, 3]]
# b=list(map(lambda x_y_z: (process(x_y_z[0]),x_y_z[1],x_y_z[2]),a))
# print(b)
# c=list(map(lambda x : 10 if x[0][1] else x[1],b))
# print(c)

# path = "./test_list.pkl"
#
# list = pkl.load(open(path, 'rb'))
# print(list)

# while 1:
#     str = input("input your commnet:")
#     if(str=="quit"):
#         break
#     else:
#         print(str)a

def func():
    print("from func\n")

if __name__ == "__main__":
    # print("from main")
    func()