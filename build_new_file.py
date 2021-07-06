import spacy
import os
from tqdm import tqdm
import pickle as pkl

MAX_VOCAB_SIZE = 25_000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
vocab_path = "E:\\PycharmProjects\\pytorch_practice\\vocab.pkl"
symbol_del = {",", ".", "\"", "-", "/><br", "(", ")", "!", "\'", "?", "...", ":", "<", "br", ";", "*", "/", "--", "&"}
output_file_path = "final_train.txt"
label_name = {"neg": 0, "pos": 1}

if __name__ == "__main__":
    BASDIR = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.abspath(os.path.join(BASDIR, ".data", "imdb", "aclImdb", "train"))
    for root, dir, files in os.walk(train_path):
        print("under the {} number of files is {}".format(root, len(files)))
    vocab = pkl.load(open(vocab_path, 'rb'))
    nlp = spacy.load('en_core_web_sm')
    nlp.Defaults.stop_words |= symbol_del
    token_count = 0

    with open(output_file_path, 'w', encoding='UTF-8') as w:
        for root, dir, files in os.walk(train_path):
            if root.endswith("neg") or root.endswith("pos"):
                for file in tqdm(files):
                    token_count = 0
                    with open(os.path.join(root, file), 'r', encoding='UTF-8') as f:
                        for line in f:
                            for word in nlp(line):
                                if not nlp.vocab[word.text].is_stop:
                                    w.write(word.text + " ")
                                    token_count += 1
                    w.write("\t"+str(token_count)+"\t"+str(label_name.get(root[-3:])))
                    w.write("\n")
