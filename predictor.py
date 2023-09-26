from .models import keyword_model
from .datasets import dataset_tag
from . import configs
from . import utils
from . import constant
import jieba
from jieba import posseg as pseg
import re
import numpy as np
import tensorflow as tf
import argparse
import os



class text_processor():
    def __init__(self,dataset_path="../test",max_len=70):
        self.max_len = max_len
        self.ds = dataset_tag(dataset_path,max_len=max_len,train=False)
        self.load_trigger_word()

    def load_trigger_word(self):
        content = utils.load_file(self.ds.trigger_words_file)
        self.trigger_words = [w.strip() for w in content]
        self.trigger_words_set = set(self.trigger_words)
        self.add_trgger_words_to_jieba()

    def is_trigger(self, sentences):
        ret = []
        trigger_words=[]
        if type(sentences) == str:
            sentences = [sentences]
        for sentence in sentences:
            if type(sentence) == list:
                sentence = "".join(sentence)
            for words in self.trigger_words:
                if words in sentence:
                    ret.append(1)
                    trigger_words.append(words)
                    break
            else:
                trigger_words.append("")
                ret.append(0)
        return ret, trigger_words

    def add_trgger_words_to_jieba(self):
        trigger_tag = "kw"
        max_freq = 1000000000
        for word in self.trigger_words:
            jieba.add_word(word, max_freq, tag=trigger_tag)
        print("add {num} words into jieba dict".format(num=len(self.trigger_words)))

    def my_tag_cut(self,text):
        ret = list(pseg.cut(text))
        words = [list(r)[0] for r in ret]  # r type is pair
        tags = [list(r)[1] for r in ret]
        return words, tags

    def tag_cut(self,texts,train=False):
        words, tags = [], []
        for t in texts:
            ret_w, ret_t = self.my_tag_cut(t)
            words.append(ret_w)
            tags.append(ret_t)
        return np.array(words), np.array(tags)

    def delete_char(self,texts):
        ret_texts = []
        for t in texts:
            t = re.sub('[a-zA-Z]', '', t)
            for c in constant.OMIT_CHAR_LIST:
                t = t.replace(c, "")
            ret_texts.append(t)
        return ret_texts

    def pre_process_tag(self, texts):
        if type(texts) == str:
            texts = [texts]
        texts = self.delete_char(texts)
        text, tags = self.tag_cut(texts)
        return text, tags

    def pre_process_padding(self, text, tags):
        padding_text = utils.padding(text, self.ds.SOS, self.ds.PAD, self.ds.EOS, self.max_len)
        padding_tags = utils.padding(tags, self.ds.SOS, self.ds.PAD, self.ds.EOS, self.max_len)
        text = utils.text2id(padding_text, self.ds.vocabs, self.ds.UNK)
        tags = utils.text2id(padding_tags, self.ds.tag_vocab, self.ds.UNK)
        keyword_vector = self.ds.gen_keyword_vector_action(tags)
        return text, tags, keyword_vector

    def process(self,texts):
        text, tags = self.pre_process_tag(texts)
        text, tags, keyword_vector = self.pre_process_padding(text, tags)
        return text, tags, keyword_vector
    @staticmethod
    def restore_label(labels, probs, id_list, trigger_words_lsit):
        new_labels, new_probs = [],[]
        new_trigger_words_list = []
        start, end = 0,0
        while start < len(id_list):
            while end < len(id_list) and id_list[end] == id_list[start]:
                end += 1
            if 1 in labels[start:end]:
                new_labels.append(1)
                new_probs.append(max(probs[start:end]))
                new_trigger_words_list.append([ keyword for keyword in trigger_words_lsit[start:end] if keyword!=""])
            else:
                new_labels.append(0)
                new_probs.append(min(probs[start:end]))
                new_trigger_words_list.append([])
            start = end
        assert len(new_labels) == len(new_probs) == len(set(id_list))
        return new_labels, new_probs, new_trigger_words_list


    def cut_text(self, texts, tags_list):
        """
        cut texts by max_len
        :param texts:
        :param tags:
        :return:
        """
        cutted_texts, cutted_tags = [], []
        cutted_id_list = []
        for i, (tokens, tags) in enumerate(zip(texts, tags_list)):
            empty_flag = False
            if len(tokens)==0:
                empty_flag = True
            while len(tokens)>0 and len(tokens) > self.max_len:
                assert len(tokens) == len(tags)
                cutted_id_list.append(i)
                cutted_texts.append(tokens[:self.max_len])
                cutted_tags.append(tags[:self.max_len])
                tokens = tokens[self.max_len:]
                tags = tags[self.max_len:]
            else:
                if len(tokens)>0 or empty_flag:
                    cutted_id_list.append(i)
                    cutted_texts.append(tokens)
                    cutted_tags.append(tags)
        return cutted_texts, cutted_tags, cutted_id_list


class Predictor:
    def __init__(self,path):
        model_path = path + os.sep + "model"
        hpath = model_path + os.sep + "hparams.ini"
        ckpt_path = model_path + os.sep + "best_f1_model"
        dataset_path = path + os.sep + "dataset"
        _configs = configs.configs(hpath)
        _configs.add_params("dataset_path", dataset_path)
        _configs.add_params("ckpt_path", ckpt_path)

        self.text_processor = text_processor(_configs.dataset_path,int(_configs.sen_len))
        tf.reset_default_graph()
        _configs.training = False
        _configs.add_params("tag_vocab_size",self.text_processor.ds.tag_vocab_len)
        _configs.add_params("vocab_size",self.text_processor.ds.vocabs_len)
        self.ckpt_path = _configs.ckpt_path
        self.model = keyword_model(_configs)
        self.sess = self.restore_model()
        self.output = tf.nn.softmax(self.model.logits)

    def restore_model(self):
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess,self.ckpt_path)
        return sess

    def predict(self, sentences, batch_size=30):
        texts, tags = self.text_processor.pre_process_tag(sentences)
        cutted_texts, cutted_tags, cutted_id_list = self.text_processor.cut_text(texts, tags)
        has_trigger_label, trigger_words_list = self.text_processor.is_trigger(cutted_texts)
        has_trigger_label = np.array(has_trigger_label)
        # 过滤不含关键词的句子，只把包含关键词的切分后的句子送到模型进行预测
        filted_cutted_texts = []
        filted_cutted_texts_id = []
        filted_cutted_tags = []
        cutted_texts_labels = [0]*len(cutted_texts)
        cutted_texts_probs = [0]*len(cutted_texts)
        for i,tflag in enumerate(has_trigger_label):
            if tflag == 1:
                filted_cutted_texts.append(cutted_texts[i])
                filted_cutted_tags.append(cutted_tags[i])
                filted_cutted_texts_id.append(i)
        cutted_texts = filted_cutted_texts
        cutted_tags = filted_cutted_tags
        padding_texts, padding_tags, keyword_vector = self.text_processor.pre_process_padding(cutted_texts, cutted_tags)
        probs, labels = [],[]
        batch = 0
        while batch_size*batch < len(padding_texts):
            start, end = batch*batch_size,min(batch_size*(batch+1),len(padding_texts))
            input_texts = padding_texts[start:end]
            input_tags = padding_tags[start: end]
            input_keyword_vector = keyword_vector[start:end]
            _l, _p = self._predict(input_texts, input_tags, input_keyword_vector)
            labels.extend(_l)
            probs.extend(_p)
            batch += 1
        for i, cid in enumerate(filted_cutted_texts_id):
            cutted_texts_labels[cid] = labels[i]
            cutted_texts_probs[cid] = probs[i]
        labels, probs = cutted_texts_labels, cutted_texts_probs
        # no valid input , return empty lists
        if len(labels)<=0 or len(has_trigger_label)<=0:
            return [], [], []
        labels = labels&has_trigger_label
        probs = probs * has_trigger_label
        labels, probs, trigger_words_list = self.text_processor.restore_label(labels, probs, cutted_id_list, trigger_words_list)
        return labels, probs, trigger_words_list

    def _predict(self, input_x, input_tags, input_keyword_vector):
        output = self.sess.run(self.output, feed_dict={self.model.input_x: input_x,
                                                       self.model.keep_prob: 1.0,
                                                       self.model.input_tag_x: input_tags,
                                                       self.model.keyword_v: input_keyword_vector
                                                       })
        if len(output.shape) == 1:
            output = np.array([output])
        labels = np.argmax(output, 1)
        labels_prob = np.array([probs[1] for index, probs in zip(labels, output)])
        return labels, labels_prob

    def basic_predict(self,text):
        input_x,input_tags,input_keyword_vector = self.text_processor.process(text)
        labels, labels_prob = self._predict(input_x, input_tags, input_keyword_vector)
        return labels, labels_prob


if __name__ == "__main__":

    #demo
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/workspace/tmp/package_model_praise")
    args = parser.parse_args()
    pred = Predictor(args.model_path)
    print("test.....")
    test_sen=["你要好好记笔记很棒","你要好好读书","你要好好玩笔记本电脑","我只是用来加长文本的内容能不能识别笔记得记啊我的天南"
                                              "快点超过吧我只是用来加长文本的内容能不能识别笔记得"
                                              "我只是用来加长文本的内容能不能识别笔记得"
                                              "我只是用来加长文本的内容能不能识别笔记得"
                                              "我只是用来加长文本的内容能不能识别笔记得"
                                              "我只是用来加长文本的内容能不能识别笔记得"
                                              "我只是用来加长文本的内容能不能识别笔记得","好记性不如烂笔头","这句话要记住"]
    for t in  test_sen:
        print("====================")
        print("input sentence: "+t)
        label,prob,_= pred.basic_predict(t)
        print("label:{label},prob:{prob}".format(label=label[0],prob=prob[0]))






