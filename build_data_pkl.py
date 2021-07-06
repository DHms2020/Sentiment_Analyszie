import spacy
import os
from tqdm import tqdm
import pickle as pkl

MAX_VOCAB_SIZE = 25_000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
symbol_del = {",", ".", "\"", "-", "/><br", "(", ")", "!", "\'", "?", "...", ":", "<", "br", ";", "*", "/", "--", "&"}
output_file_path_train = "train_data.pkl"
output_file_path_test = "test_data.pkl"
label_name = {"neg": 0, "pos": 1}

if __name__ == "__main__":
    BASDIR = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.abspath(os.path.join(BASDIR, ".data", "imdb", "aclImdb", "train"))
    test_path = os.path.abspath(os.path.join(BASDIR, ".data", "imdb", "aclImdb", "test"))
    for root, dir, files in os.walk(train_path):
        print("under the {} number of files is {}".format(root, len(files)))
    vocab = pkl.load(open((os.path.abspath(os.path.join(".", "vocab.pkl"))), 'rb'))
    nlp = spacy.load('en_core_web_sm')
    nlp.Defaults.stop_words |= symbol_del
    train_data = []
    for root, dir, files in os.walk(train_path):
        if root.endswith("neg") or root.endswith("pos"):
            for file in tqdm(files):
                with open(os.path.join(root, file), 'r', encoding='UTF-8') as f:
                    label = label_name.get(root[-3:])
                    text_content_id = []
                    for line in f:
                        for word in nlp(line):
                            if not nlp.vocab[word.text].is_stop:
                                text_content_id.append(vocab.get(word.text, vocab.get(UNK)))
                train_data.append((float(label), text_content_id))
    pkl.dump(train_data, open(output_file_path_train, 'wb'))

    test_data = []
    for root, dir, files in os.walk(test_path):
        if root.endswith("neg") or root.endswith("pos"):
            for file in tqdm(files):
                with open(os.path.join(root, file), 'r', encoding='UTF-8') as f:
                    label = label_name.get(root[-3:])
                    text_content_id = []
                    for line in f:
                        for word in nlp(line):
                            if not nlp.vocab[word.text].is_stop:
                                text_content_id.append(vocab.get(word.text, vocab.get(UNK)))
                test_data.append((float(label), text_content_id))
    pkl.dump(test_data, open(output_file_path_test, 'wb'))
