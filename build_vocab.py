# import torch
# from torchtext import data
# from torchtext import datasets
import spacy
import os
from tqdm import tqdm
import pickle as pkl

MAX_VOCAB_SIZE = 25_000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
vocab_path = "E:\\PycharmProjects\\pytorch_practice\\vocab.pkl"
symbol_del = {",", ".", "\"", "-", "/><br", "(", ")", "!", "\'", "?", "...", ":", "<", "br", ";", "*", "/", "--", "&"}


# SEED = 1234
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
#
# TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
# LABEL = data.LabelField(dtype=torch.float)
#
# train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
# print(f'Number of training examples: {len(train_data)}')
# print(f'Number of testing examples: {len(test_data)}')

def build_vocab(dir_path, max_size, min_freq=1):
    vocab_dict_freq = {}
    nlp = spacy.load('en_core_web_sm')
    nlp.Defaults.stop_words |= symbol_del
    for root, _, files in os.walk(dir_path):
        if root == os.path.join(dir_path, "neg") or root == os.path.join(dir_path, "pos"):
            for file in tqdm(files):
                with open(os.path.join(root, file), 'r', encoding='UTF-8') as f:
                    for line in f:
                        for word in nlp(line):
                            if not nlp.vocab[word.text].is_stop:
                                vocab_dict_freq[word.text] = vocab_dict_freq.get(word.text, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dict_freq.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                 :max_size]
    # print(len(vocab_list))
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


if __name__ == "__main__":
    BASDIR = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.abspath(os.path.join(BASDIR, ".data", "imdb", "aclImdb", "train"))
    test_path = os.path.abspath(os.path.join(BASDIR, ".data", "imdb", "aclImdb", "test"))
    print(train_path)
    for root, dir, files in os.walk(train_path):
        print("under the {} number of files is {}".format(root, len(files)))
    # print(MAX_VOCAB_SIZE)

    if os.path.exists(vocab_path):
        vocab = pkl.load(open(vocab_path, 'rb'))
    else:
        vocab = build_vocab(train_path, MAX_VOCAB_SIZE)
        pkl.dump(vocab, open(vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")
