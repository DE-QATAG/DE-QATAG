#!/usr/bin/env python
import tensorflow as tf
from data import load_dict,BUCKET,pad_sequences,process_label_pattern
import os
import modeling
class SeqModel(object):
    def __init__(self, args, logger,hvd=False):
        self.args=args
        self.logger = logger 
        self.hvd=hvd
        self.optimizer = args.optimizer
        self.num_hidden_layers = args.layer_depth
        self.embedding_dim = args.embedding_dim
        self.bucket = BUCKET
        self.ngram_dim=args.ngram_dim
        self.tag2id, self.id2tag = load_dict(args.tag_to_id_path,args.encoding)
        self.num_tags = 3 # len(self.tag2id)
        self.word2id, self.id2word = load_dict(args.word_to_id_path,args.encoding)
        self.logger.info("tag2id size: %d" % self.num_tags)
        self.logger.info("word2id size: %d" % len(self.word2id))
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
        self.lambda4 = args.lambda4
    def build_bert_graph(self,bert_base_dir):
        self.add_placeholders()
        self.bert_layer_op(bert_base_dir)
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def build_cnn_ngram_graph(self,num_filters,filter_width=3,embeddings=None,bigra=None,trigram=None):
        self.add_placeholders_ngram()
        self.lookup_layer_ngram(embeddings,bigra,trigram)
        self.cnn_layer_ngram(num_filters,filter_width)
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def trainstep_op(self,clip_nom=1):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            learning_rate = self.lr_pl
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            if self.args.use_hvd:
                self.logger.info("==== use DistributedOptimizer")
                optim= self.hvd.DistributedOptimizer(optim)
            tvars=tf.trainable_variables()
            grads_and_vars = optim.compute_gradients(self.loss,tvars)
            grads,_ = tf.clip_by_global_norm([k for k,v in grads_and_vars], clip_norm=clip_nom)
            self.train_op = optim.apply_gradients(zip(grads,tvars), global_step=self.global_step)
            #self.train_op = optim.apply_gradients(grads_and_vars, global_step=self.global_step)
    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_placeholders_ngram(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.before_bigram_ids = tf.placeholder(tf.int32, shape=[None, None], name="before_bigram_ids")
        self.after_bigram_ids = tf.placeholder(tf.int32, shape=[None, None], name="after_bigram_ids")
        self.before_trigram_ids = tf.placeholder(tf.int32, shape=[None, None], name="before_trigram_ids")
        self.after_trigram_ids = tf.placeholder(tf.int32, shape=[None, None], name="after_trigram_ids")
    
    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.seg_ids = tf.placeholder(tf.int32, shape=[None, None], name="seg_ids")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.querys = tf.placeholder(tf.int32, shape=[None, None], name="labels_ori")
        self.cls = tf.placeholder(tf.int32, shape=[None], name="cls_label")

    def get_feed_dict(self, seqs, labels, segments, lr, dropout, ngrams=None):
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)
        seg_ids, _ = pad_sequences(segments, pad_mark=0)
        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list,
                     self.seg_ids: seg_ids}
        if labels is not None:
            labels_, query_, cls_ = process_label_pattern(labels)
            feed_dict[self.labels] = labels_
            feed_dict[self.querys] = query_
            feed_dict[self.cls] = cls_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout
        if ngrams is not None:
            before_bigrams_, _ = pad_sequences(ngrams[0], pad_mark=0)
            feed_dict[self.before_bigram_ids] = before_bigrams_
            after_bigrams_, _ = pad_sequences(ngrams[1], pad_mark=0)
            feed_dict[self.after_bigram_ids] = after_bigrams_
            before_trigrams_, _ = pad_sequences(ngrams[2], pad_mark=0)
            feed_dict[self.before_trigram_ids] = before_trigrams_
            after_trigrams_, _ = pad_sequences(ngrams[3], pad_mark=0)
            feed_dict[self.after_trigram_ids] = after_trigrams_
        return feed_dict, seq_len_list, len(word_ids), len(word_ids[0])


    def bert_layer_op(self,bert_base_dir):
        bert_config_path = os.path.join(bert_base_dir, "bert_config.json") 
        self.bert_config = modeling.BertConfig.from_json_file(bert_config_path)
        self.bert_config.num_hidden_layers = self.num_hidden_layers
        self.input_mask = tf.sequence_mask(self.sequence_lengths, dtype=tf.int32)
        self.logger.info(self.bert_config.to_json_string())
        self.model = modeling.BertModel(config=self.bert_config, dropout_rate=self.dropout_pl, 
                                        input_ids=self.word_ids, input_mask=self.input_mask, token_type_ids=self.seg_ids,
                                        use_one_hot_embeddings=False)
        if self.args.mode == "train":
            checkpoint_file = os.path.join(bert_base_dir, "bert_model.ckpt")
            if checkpoint_file:
                assignment_map, initialized_variable_names = modeling.get_assigment_map_from_checkpoint(
                    tf.trainable_variables(), checkpoint_file)
                tf.train.init_from_checkpoint(checkpoint_file, assignment_map)
                self.logger.info("load checkpoint_file successfully!!")           
    
        self.embedding_output = self.model.get_embedding_output()
        self.sequence_outputs = self.model.get_sequence_output()

        with tf.variable_scope("bert-encoder"):
            out_shape = self.sequence_outputs.shape.as_list()
            last_channel_size = out_shape[-1]

            o_w = tf.get_variable("logits-w", shape=[last_channel_size, self.num_tags], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            o_b = tf.get_variable("logits-b", initializer=tf.constant(0.0, shape=[self.num_tags]))
            output_reshape = tf.reshape(self.sequence_outputs, [-1, last_channel_size])
            pred = tf.nn.xw_plus_b(output_reshape, o_w, o_b)
            s = tf.shape(self.sequence_outputs)
            logits = tf.reshape(pred, [-1, s[1], self.num_tags])
            self.logits = logits
            tf.add_to_collection("logits", self.logits)

            cls_input_reshape = self.sequence_outputs[:, 0, :]
            # cls_input_reshape = tf.reshape(cls_output, [-1, last_channel_size])
            c_w = tf.get_variable("classes-w", shape=[last_channel_size, 2], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            c_b = tf.get_variable("classes-b", initializer=tf.constant(0.0, shape=[2]))
            classes = tf.nn.xw_plus_b(cls_input_reshape, c_w, c_b)
            self.classes = classes
            tf.add_to_collection("classes", self.classes)


    def lookup_layer_ngram(self,embeddings=None,bigram=None,trigram=None):
        feature_embs = []
        with tf.variable_scope("words"):
            if embeddings:
                _word_embeddings = tf.Variable(embeddings, dtype=tf.float32, name="word_embeddings")
            else:
                _word_embeddings = tf.get_variable('word_embeddings', shape=[len(self.word2id), self.embedding_dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())

            if bigram:
                _bigram_embeddings = tf.Variable(bigram, dtype=tf.float32, name="bigram_embeddings")
            else:
                _bigram_embeddings = tf.get_variable('bigram_embeddings', shape=[self.bucket, self.ngram_dim],
                                                     initializer=tf.contrib.layers.xavier_initializer())
            if trigram:
                _trigram_embeddings = tf.Variable(trigram, dtype=tf.float32, name="trigram_embeddings")
            else:
                _trigram_embeddings = tf.get_variable('trigram_embeddings', shape=[self.bucket, self.ngram_dim],
                                                      initializer=tf.contrib.layers.xavier_initializer())

            # word embedding
            self.word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=self.word_ids,
                                                          name="word_embeddings")
            feature_embs.append(self.word_embeddings)
            # bigram
            before_bigram_embeddings = tf.nn.embedding_lookup(params=_bigram_embeddings,
                                                              ids=self.before_bigram_ids)
            after_bigram_embeddings = tf.nn.embedding_lookup(params=_bigram_embeddings, ids=self.after_bigram_ids)
            feature_embs.append(before_bigram_embeddings)
            feature_embs.append(after_bigram_embeddings)
            # trigram
            before_trigram_embeddings = tf.nn.embedding_lookup(params=_trigram_embeddings,
                                                               ids=self.before_trigram_ids)
            after_trigram_embeddings = tf.nn.embedding_lookup(params=_trigram_embeddings,
                                                              ids=self.after_trigram_ids)
            feature_embs.append(before_trigram_embeddings)
            feature_embs.append(after_trigram_embeddings)
            self.embs_len = len(feature_embs)
        
        feature_concat = tf.concat(feature_embs, -1)
        self.logger.info("feature_concat shape: {}".format(feature_concat.shape))
        zero_one_mask = tf.sequence_mask(self.sequence_lengths, dtype=tf.float32)
        zero_one_pad = tf.expand_dims(zero_one_mask, -1)
        no_zero_embeddings = tf.multiply(feature_concat, zero_one_pad)        
        self.zero_one_pad = tf.expand_dims(zero_one_pad, 1)
        
        self.uni_bi_tri_embeddings = tf.nn.dropout(no_zero_embeddings, 1-self.dropout_pl)
        self.logger.info("no_zero_embeddings shape: {}".format(no_zero_embeddings.shape))
        self.logger.info("uni_bi_tri_embeddings shape: {}".format(self.uni_bi_tri_embeddings.shape))
        self.input_len =  (self.embs_len - 1) * self.ngram_dim  + self.embedding_dim

    def cnn_layer_ngram(self,num_filters,filter_width=3):
        self.logger.info("cnn_layer_op beigin")
        with tf.variable_scope("cnn"):
            initial_num_filters = num_filters
            self.logger.info("uni_bi_tri_embeddings shape: {}".format(self.uni_bi_tri_embeddings.shape))
            input_feats_expanded = tf.expand_dims(self.uni_bi_tri_embeddings, 1)

            last_channel_size = self.input_len
            lastoutput = input_feats_expanded
            features = []
            self.conv = []
            features.append(input_feats_expanded)
            self.conv.append(())
            for i in range(0, self.num_hidden_layers):
                if i == 0:
                    width = 1
                else:
                    width = filter_width

                with tf.variable_scope("layer-%d" % i):
                    filter_shape = [1, width, last_channel_size, initial_num_filters]
                    w = tf.get_variable("w", shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.get_variable("b", initializer=tf.constant(0.0, shape=[initial_num_filters]))
                    self.logger.info("lastoutput shape: {}".format(lastoutput.shape))
                    self.logger.info("w shape: {}".format(w.shape))
                    conv_op = tf.nn.conv2d(lastoutput, w, strides=[1,1,1,1], padding="SAME",
                                           use_cudnn_on_gpu=True, data_format='NHWC', name="conv-layer-%d" % i)
                    conv_b = tf.nn.bias_add(conv_op, b)
                    conv_relu = tf.nn.relu(conv_b)
                    no_zero_conv_relu = tf.multiply(conv_relu, self.zero_one_pad)
                    # update
                    lastoutput = no_zero_conv_relu
                    self.conv.append((conv_b, w, b))
                    features.append(no_zero_conv_relu)
                    last_channel_size = initial_num_filters

                    '''
                    # dense cnn
                    lastoutput = tf.concat([lastoutput, conv_relu], 3)  # [batch, 1, length, num_filer + last_output_channel_num]
                    last_channel_size += initial_num_filters
                    '''
            # self.cnn_output = lastoutput
            self.features = features
            self.cnn_output = tf.concat(features, 3)
            last_channel_size = self.input_len + initial_num_filters * self.num_hidden_layers

            o_w = tf.get_variable("logits-w", shape=[last_channel_size, self.num_tags], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            o_b = tf.get_variable("logits-b", initializer=tf.constant(0.0, shape=[self.num_tags]))
            output_squeeze = tf.squeeze(self.cnn_output, [1])
            output_reshape = tf.reshape(output_squeeze, [-1, last_channel_size])
            self.logger.info("output_reshape shape: {}".format(output_reshape.shape))
            self.logger.info("o_w shape: {}".format(o_w.shape))
            self.logger.info("o_b shape: {}".format(o_b.shape))
            pred = tf.nn.xw_plus_b(output_reshape, o_w, o_b)
            s = tf.shape(output_squeeze)
            logits = tf.reshape(pred, [-1, s[1], self.num_tags])
            self.logits = logits
            tf.add_to_collection("logits", self.logits)
        self.logger.info("cnn_layer_op end")

    def softmax_pred_op(self):
        self.softmax = tf.nn.softmax(self.logits)
        self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
        self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)
        tf.add_to_collection("label_softmax",self.labels_softmax_)

    def loss_op(self):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.labels)
        mask = tf.sequence_mask(self.sequence_lengths)
        losses = tf.boolean_mask(losses, mask)
        self.loss1 = tf.reduce_mean(losses)
        # loss2 desc consim
        softmax_true = tf.argmax(self.logits, axis=-1)
        softmax_false = tf.zeros_like(softmax_true)
        labels_softmax_ = tf.where(mask, softmax_true, softmax_false)
        labels_pred_ = tf.cast(labels_softmax_, dtype=tf.bool)
        labels_gold_ = tf.cast(self.labels, dtype=tf.bool)

        labels_softmax_ = tf.expand_dims(tf.cast(labels_pred_,dtype=tf.float32),-1)
        labels_softmax_ori = tf.expand_dims(tf.cast(labels_gold_,dtype=tf.float32),-1)

        input_desc = tf.multiply(labels_softmax_ori,self.sequence_outputs)
        pred_desc = tf.multiply(labels_softmax_,self.sequence_outputs)
        
        input_desc = tf.reduce_sum(input_desc,axis=1) # batch*seq_len*hidden_size --> batch*hidden_size
        pred_desc = tf.reduce_sum(pred_desc,axis=1)

        input_elem = tf.sqrt(tf.reduce_sum(tf.multiply(input_desc,input_desc),-1))  # batch
        pred_elem = tf.sqrt(tf.reduce_sum(tf.multiply(pred_desc,pred_desc), -1))

        desc_cos_matrix = tf.multiply(input_desc,pred_desc)
        desc_cos_vec = tf.reduce_sum(desc_cos_matrix,axis=-1)       # batch

        desc_cos_score = tf.div(desc_cos_vec, input_elem * pred_elem + 1e-8)    # batch
        # desc_cos_vec_sum = tf.reduce_sum(desc_cos_score,axis=-1)
        loss2 = tf.log(2/(desc_cos_score+1+1e-8))
        self.loss2 = tf.reduce_mean(loss2)

        # loss4
        tmp_one = tf.ones_like(self.labels,dtype=tf.int32)
        tmp_zero = tf.zeros_like(self.labels,dtype=tf.int32)
        labels_diff = tf.where(tf.equal(labels_gold_, labels_pred_), tmp_zero, tmp_one) # batch*seq_len
        cls_true = tf.ones_like(self.cls,dtype=tf.int32)
        cls_false = tf.zeros_like(self.cls,dtype=tf.int32)
        cls_bool = tf.cast(tf.reduce_sum(labels_diff, axis=-1), dtype=tf.bool)
        self.cls_pair = tf.where(cls_bool, cls_false, cls_true) # >1:0,=0:1
        label_name = tf.expand_dims(tf.cast(self.querys, dtype=tf.float32), -1)
        query_name = tf.reduce_mean(tf.multiply(label_name, self.sequence_outputs), axis=1)

        query_desc =  tf.reduce_mean(tf.multiply(labels_softmax_,self.sequence_outputs), axis=1)

        cls_input = tf.concat([query_name, query_desc], 1)    # batch*2hidden
        out_shape = self.sequence_outputs.shape.as_list()
        last_channel_size = out_shape[-1]
        # pair cls
        p_w = tf.get_variable("pair-w", shape=[2*last_channel_size, 2], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        p_b = tf.get_variable("pair-b", initializer=tf.constant(0.0, shape=[2]))
        self.paires = tf.nn.xw_plus_b(cls_input, p_w, p_b)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.paires, labels=self.cls_pair)
        self.loss4 = tf.reduce_mean(losses)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.classes, labels=self.cls)
        self.loss3 = tf.reduce_mean(losses)

        self.loss = self.lambda1 * self.loss1 + self.lambda2 * self.loss2 + self.lambda3 * self.loss3 + self.lambda4 * self.loss4






