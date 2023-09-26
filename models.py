from __future__ import print_function
import numpy as np
import tensorflow as tf

class keyword_model():
    def __init__(self, configs):

        self.sen_len = int(configs.sen_len)
        self.tag_vocab_size = int(configs.tag_vocab_size)
        self.vocab_size = int(configs.vocab_size)
        self.embedding_size = int(configs.embedding_size)
        self.early_stop_num = 50
        self.batch_size = int(configs.batch_size)
        self.class_num = int(configs.class_num)
        self.drop_rate = float(configs.drop_rate)
        self.l2_lambda = float(configs.l2_lambda)
        self.lr_decay_step = int(configs.lr_dacay_step)
        self.lr_decay_rate = float(configs.lr_decay_rate)
        self.h = int(configs.head_num)
        self.training = bool(configs.training)
        self.build_placeholder()
        self.build_embedding()
        self.construct_model()

    def build_placeholder(self):
        # 定义palceholder
        self.input_x = tf.placeholder(tf.int32, shape=(None, self.sen_len), name="input_x")
        self.input_y = tf.placeholder(tf.int64, shape=(None), name="input_y")
        self.keep_prob = tf.placeholder("float")
        self.input_tag_x = tf.placeholder(tf.int32,shape=(None,self.sen_len),name="input_x_tag")
        self.keyword_v = tf.placeholder(tf.float32,shape=(None,self.sen_len),name="input_kw_v")


    # 定义embedding
    # 预先训练的embedding 和 新训练embedding
    def build_embedding(self):
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            self.train_embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, mask_zero=True,
                                                             input_length=self.sen_len,name="word_Embedding")

            self.tag_embedding = tf.keras.layers.Embedding(self.tag_vocab_size,self.embedding_size,mask_zero=True,
                                                           input_length=self.sen_len,name="tag_Embedding")

    def positional_encoding(self,
                            inputs,
                            maxlen,
                            masking=True,
                            scope="positional_encoding"):

        E = inputs.get_shape().as_list()[-1]  # static
        N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # position indices
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
                for pos in range(maxlen)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

            # lookup
            outputs = tf.nn.embedding_lookup(position_enc, position_ind)

            # masks
            if masking:
                outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)
            return tf.to_float(outputs)

    def embed_x(self, x,x_tag):
        train_embed = self.train_embedding(x)
        tag_embed = self.tag_embedding(x_tag)
        embed = train_embed*tf.nn.sigmoid(tag_embed)
        return embed


    def construct_model(self):
        _embed_x_ = self.embed_x(self.input_x,self.input_tag_x)  # B X L X2*embed_size
        x = self.multi_head_att(_embed_x_
                                , scope="multi_head_att_1")  # B X L X embed_size
        self.logits = self.cal_logit(x)


    def multi_head_att(self, x, scope="multi_head_att"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(x, self.embedding_size, use_bias=False)

            # split and concat
            x_ = tf.concat(tf.split(x, self.h, axis=2), axis=0)

            # attention
            outputs = self.self_att(x_)

            # restore
            outputs = tf.concat(tf.split(outputs, self.h, axis=0), axis=2)
            outputs = self.ln(outputs)
            o_1 = tf.layers.dense(outputs, self.embedding_size / 2, activation=tf.nn.relu)
            o_2 = tf.layers.dense(o_1, self.embedding_size)
            outputs += o_2
            return self.ln(outputs)

    def ln(self, inputs, epsilon=1e-8, scope="ln"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def self_att(self, x):
        with tf.variable_scope("self_att", reuse=tf.AUTO_REUSE):
            scale = np.math.sqrt(self.embedding_size / self.h)  # scale = np.math.sqrt(2*self.embedding_size)
            scores = tf.nn.softmax(tf.matmul(x, tf.transpose(x, perm=[0, 2, 1])) / scale)  # B X L X L
            scores = tf.transpose(scores, [0, 2, 1])
            self.self_att_scores = scores
            scores = tf.layers.dropout(scores, rate=self.drop_rate, training=self.training)
            outputs = tf.matmul(scores, x)  # B X L X  embed_size
            return outputs



    def cal_logit(self, x):
        with tf.variable_scope("cal_logit", reuse=tf.AUTO_REUSE):
            #             print(x.shape)

            # vector in keyword position
            kw_vecotrs = tf.expand_dims(self.keyword_v,axis=2) #B X S X 1
            kw_vecotrs = tf.matmul(tf.transpose(kw_vecotrs,[0,2,1]),x) # B X 1 X E_d
            #vector in start position
            x = tf.slice(x, [0, 0, 0], [-1, 1, -1]) #B X 1 X E_d
            x = tf.concat([x,kw_vecotrs],axis=2)
            output = tf.layers.dense(x,self.class_num,use_bias=False)
            return tf.squeeze(output)


    def train_op(self):
        with tf.variable_scope("train_op", reuse=tf.AUTO_REUSE):
            reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.l2_lambda),
                                                         tf.trainable_variables())
            train_loss = tf.losses.sparse_softmax_cross_entropy(self.input_y, self.logits)+reg
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(0.1, global_step,120, 0.98, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.01)
            train_op = optimizer.minimize(train_loss, global_step=global_step)

            tf.summary.scalar("train_loss",train_loss)
            tf.summary.scalar("global_step",global_step)
            tf.summary.scalar("lr",optimizer._lr)
            # tf.summary.scalar("lr", self.optimizer._learning_rate)
            train_summaries = tf.summary.merge_all()
            return train_op,train_loss,train_summaries,global_step

    def get_metrics(self, predicted, actual):
        TP = tf.count_nonzero(predicted * actual)
        FP = tf.count_nonzero(predicted * (actual - 1))
        FN = tf.count_nonzero((predicted - 1) * actual)
        precision = tf.divide(TP, (TP + FP))
        recall = tf.divide(TP, (TP + FN))
        f1 = tf.divide(2 * precision * recall, (precision + recall))
        return precision, recall, f1


    def eval_op(self,scope_name = "eval_op"):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.l2_lambda),
                                                         tf.trainable_variables())
            eval_loss = tf.losses.sparse_softmax_cross_entropy(self.input_y, self.logits) +reg
            pred_y = tf.argmax(self.logits,1)
            correct_prediction = tf.equal(pred_y, tf.cast(self.input_y,dtype=tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            global_step = tf.train.get_or_create_global_step()
            precision, recall, f1 = self.get_metrics(pred_y, tf.cast(self.input_y,dtype=tf.int64))
            tf.summary.scalar("valid_loss", eval_loss)
            tf.summary.scalar("global_step", global_step)
            tf.summary.scalar("valid_f1", f1)
            tf.summary.scalar("valid_precision", precision)
            tf.summary.scalar("valid_recall", recall)
            tf.summary.scalar("valid_acc", accuracy)
            eval_summaries = tf.summary.merge_all()
            return eval_loss,eval_summaries







