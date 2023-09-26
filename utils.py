import codecs
import numpy as np
import random
import tensorflow as tf
import jieba


def load_file(file_path):
    with codecs.open(file_path,'r',encoding='utf-8') as f:
        content = f.readlines()
    return content


def split_label_text(content):
    labels, texts = [],[]
    for l in content:
        label, text = l.strip().split("\t")
        labels.append(int(label))
        texts.append(text)
    return np.array(labels),np.array(texts)


def text2id(texts,vocabs,unk):
    def word2Id(word):
        if word not in vocabs:
            return vocabs[unk]
        return vocabs[word]

    ret = []
    for s in texts:
        tmp_ids = []
        for w in s:
            if w != unk:
                tmp_ids.append(word2Id(w))
            else:
                tmp_ids.append(random.randint(4,len(vocabs)))  #随机向量 oov
        ret.append(tmp_ids)
    return np.array(ret)


def padding(texts, SOS, PAD, EOS, max_len=50):
    ret = []
    for s in texts:
        s = s[:min(max_len - 2, len(s))]
        pad_list = [PAD] * max(max_len - 2 - len(s), 0)
        tmp_sen = [SOS]
        tmp_sen.extend(s)
        tmp_sen.append(EOS)
        tmp_sen.extend(pad_list)
        ret.append(tmp_sen)
        assert len(tmp_sen) == max_len
    return np.array(ret)


def tokenize(texts, mode="word"):
    if mode not in ["word","char"]:
        print("tokenize failed!, only support [word] or [char] level!")
        return
    if mode=="word":
        texts = np.array([ list(jieba.cut(t)) for t in texts])
    if mode=="char":
        texts = np.array([ list(t) for t in texts])
    return texts


def save_variable_specs(fpath):
    '''Saves information about variables such as
    their name, shape, and total parameter number
    fpath: string. output file path
    Writes
    a text file named fpath.
    '''
    def _get_size(shp):
        '''Gets size of tensor shape
        shp: TensorShape
        Returns
        size
        '''
        size = 1
        for d in range(len(shp)):
            size *=shp[d]
        return size

    params, num_params = [], 0
    for v in tf.global_variables():
        params.append("{}==={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    print("num_params: ", num_params)
    with open(fpath, 'w') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))
    print("Variables info has been saved.")


def np_output_2_pred(output):
    return np.argmax(output,1)



class Metrics():
    def __init__(self):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_acc = []
        self.max_f1 = -1
        self.max_score = []

    def initialize(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_acc = []

    def on_epoch_end(self, val_predict, val_targ, evalu=False):
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_acc = accuracy_score(val_targ, val_predict)
        if not evalu:
            self.val_f1s.append(_val_f1)
            self.val_recalls.append(_val_recall)
            self.val_precisions.append(_val_precision)
            self.val_acc.append(_val_acc)
        else:
            if _val_f1 > self.max_f1:
                self.max_f1 = _val_f1
                self.max_score = [_val_recall, _val_precision, _val_acc]

        print(" — val_f1: %f — val_precision: %f — val_recall %f" % (_val_f1, _val_precision, _val_recall))
        return _val_f1, _val_precision, _val_recall, _val_acc

