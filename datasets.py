import os
import utils
import random
import numpy as np



VOCAB_FILE_NAME = "vocab.txt"
VOCAB_TAG_FILE_NAME = "tag_vocab.txt"
TRAIN_FILE_NAME = "train.txt"
VALID_FILE_NAME = "valid.txt"
TEST_FILE_NAME = "test.txt"
TRIGGER_WORDS_FILE_NAME = "trigger_words.txt"
TRAIN_TAG_FILE_NAME = "train_tag.txt"
VALID_TAG_FILE_NAME = "valid_tag.txt"
TEST_TAG_FILE_NAME = "test_tag.txt"



class dataset():
    def __init__(self,base_path = "./",max_len=70,train=True):
        self.base_path = base_path
        self.max_len = 70
        self.PAD = "<PAD>"
        self.EOS = "<EOS>"
        self.SOS = "<SOS>"
        self.UNK = "<UNK>"
        self.define_file_path()
        if train:
            self.load_text_label()
        self.load_vocab()

    def define_file_path(self):
        self.vocab_file = self.base_path+os.sep + VOCAB_FILE_NAME
        self.vocab_tag_file = self.base_path + os.sep + VOCAB_TAG_FILE_NAME
        self.train_file = self.base_path + os.sep + TRAIN_FILE_NAME
        self.valid_file = self.base_path + os.sep + VALID_FILE_NAME
        self.test_file = self.base_path + os.sep+ TEST_FILE_NAME
        self.trigger_words_file = self.base_path + os.sep + TRIGGER_WORDS_FILE_NAME
        self.train_tag_file = self.base_path+ os.sep + TRAIN_TAG_FILE_NAME
        self.test_tag_file = self.base_path + os.sep + TEST_TAG_FILE_NAME
        self.valid_tag_file = self.base_path + os.sep + VALID_TAG_FILE_NAME

    def load_text_label(self):
        train_content = utils.load_file(self.train_file)
        valid_content = utils.load_file(self.valid_file)
        test_content = utils.load_file(self.test_file)

        self.train_labels, self.train_texts = utils.split_label_text(train_content)
        self.valid_labels, self.valid_texts = utils.split_label_text(valid_content)
        self.test_labels, self.test_texts = utils.split_label_text(test_content)


    def load_vocab(self):
        content = utils.load_file(self.vocab_file)
        content = dict([[c.split(" ")[0], i + 4] for i, c in enumerate(content)])
        content[self.PAD] = 0
        content[self.UNK] = 1
        content[self.SOS] = 2
        content[self.EOS] = 3

        self.vocabs = content
        self.vocabs_len = len(content)


    def consturct_input_action(self,texts):
        padding_texts = utils.padding(np.array([ s.split() for s in texts]),SOS=self.SOS,
                                      PAD=self.PAD,EOS=self.EOS,max_len=self.max_len)
        input_x = utils.text2id(padding_texts,self.vocabs,self.UNK)
        return input_x

    def construct_input(self):
        self.train_input_x = self.consturct_input_action(self.train_texts)
        self.test_input_x = self.consturct_input_action(self.test_texts)
        self.valid_input_x = self.consturct_input_action(self.valid_texts)


    def batch_genenrator(self,batch_size=100):
        x = self.train_input_x
        y = self.train_labels
        sample_nums = len(x)
        iter_num = int(sample_nums // batch_size)
        indexs = np.array(list(range(sample_nums)))
        random.shuffle(indexs)
        for i in range(iter_num):
            selected_indexs = indexs[i * batch_size:(i + 1) * batch_size]
            yield x[selected_indexs], y[selected_indexs]

class dataset_tag(dataset):
    def __init__(self,base_path="./",max_len=70,train=True):
        super(dataset_tag,self).__init__(base_path,max_len,train)
        self.load_tag_vocabs()
        if train:
            self.load_tag_texts()


    def load_tag_vocabs(self):
        content = utils.load_file(self.vocab_tag_file)
        tag_vocab = dict()
        tag_vocab[self.PAD] = 0
        tag_vocab[self.UNK] = 1
        tag_vocab[self.SOS] = 2
        tag_vocab[self.EOS] = 3
        id_num = 4
        for c in content:
            tag = c.strip().split()[0]
            tag_vocab[tag.strip()] = id_num
            id_num += 1
        print("tag vocab dict length {num}".format(num=id_num))
        self.tag_vocab_len = id_num
        self.tag_vocab = tag_vocab

    def load_tag_texts(self):
        train_content = utils.load_file(self.train_tag_file)
        test_content = utils.load_file(self.test_tag_file)
        valid_content = utils.load_file(self.valid_tag_file)

        self.train_tags = np.array([c.strip().split() for c in train_content])
        self.test_tags = np.array([c.strip().split() for c in test_content])
        self.valid_tags = np.array([c.strip().split() for c in valid_content])

    def construct_input(self):
        super(dataset_tag,self).construct_input()

        self.train_input_tags = self.construct_input_tag_action(self.train_tags)
        self.test_input_tags = self.construct_input_tag_action(self.test_tags)
        self.valid_input_tags = self.construct_input_tag_action(self.valid_tags)

        self.gen_keyword_vectors()


    def construct_input_tag_action(self,tags):
        padding_tags = utils.padding(tags,self.SOS,self.PAD,self.EOS,self.max_len)
        input_tags = utils.text2id(padding_tags,self.tag_vocab,self.UNK)
        return input_tags

    def gen_keyword_vector_action(self,tag_ids):
        # print("generate keyword vectors")
        kw_tag_id = self.tag_vocab["kw"]
        # print("kw id is {ID}".format(ID=kw_tag_id))
        ret = []
        for tag in tag_ids:
            row_vector = np.zeros(self.max_len,dtype=np.float)
            row_vector[np.array(tag)==kw_tag_id] = 1
            ret.append(row_vector)
        return np.array(ret)

    def gen_keyword_vectors(self):
        self.train_keyword_vector = self.gen_keyword_vector_action(self.train_input_tags)
        self.test_keyword_vector = self.gen_keyword_vector_action(self.test_input_tags)
        self.valid_keyword_vector = self.gen_keyword_vector_action(self.valid_input_tags)

    def batch_genenrator(self,batch_size=100):
        x = self.train_input_x
        y = self.train_labels
        x_tag = self.train_input_tags
        kw_v = self.train_keyword_vector
        sample_nums = len(x)
        iter_num = int(sample_nums // batch_size)
        indexs = np.array(list(range(sample_nums)))
        random.shuffle(indexs)
        for i in range(iter_num):
            selected_indexs = indexs[i * batch_size:(i + 1) * batch_size]
            # print(selected_indexs)
            yield x[selected_indexs], y[selected_indexs],x_tag[selected_indexs],kw_v[selected_indexs]

if __name__ =="__main__":
    #usage
    ds = dataset_tag("../test")
    ds.construct_input()
    # ds.batch_genenrator()
    data_generator = ds.batch_genenrator(batch_size=100)
    for x, y, x_tag, x_kw_v in data_generator:
        pass
    # pdb.set_trace()

