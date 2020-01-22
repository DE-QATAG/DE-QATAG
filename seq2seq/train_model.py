#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from data import text2batch,bert_text_to_batch,read_file,read_file_yeild,load_dict,is_dataset_tag,load_dict_ano,pad_sequences,process_label_pattern,process_label
#from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime,timedelta
import io,os,sys,time,random
from eval_util import NerEvaluator

class TrainFramework(object):
    def __init__(self, seqmodel, paths, config):
        self.model=seqmodel
        self.args = seqmodel.args
        self.encoding=self.args.encoding
        self.logger = seqmodel.logger
        self.model_global_step=seqmodel.global_step
        self.global_step=0
        self.hvd=seqmodel.hvd
        self.model_path = paths['model_path']
        if self.model.args.restore:
            self.restore_model_path = paths['restore_model_path']
        self.summary_path = paths['summary_path']
        self.result_path = paths['result_path']
        self.tag2id, self.id2tag = load_dict(self.args.tag_to_id_path,self.encoding)
        self.word2id, self.id2word = load_dict(self.args.word_to_id_path,self.encoding)
        self.dataset2flag, self.flag2dataset = load_dict_ano(self.args.dataset_to_flag_path,self.encoding)
        self.config = config
        self.batch_size = self.args.batch_size
        self.epoch_num = self.args.epoch
        self.min_epoch_num = self.args.min_epoch
        self.restore = self.args.restore
        self.dropout = self.args.dropout
        self.optimizer = self.args.optimizer
        self.lr = self.args.lr
        self.max_scores = 0.0
        self.unseccessful_step_num = 0
        self.eval_step = self.args.eval_step
        self.local_step_num = 0
        self.total_w_count = 0
        self.total_w_loss = 0
        self.total_w_loss1 = 0
        self.total_w_loss2 = 0
        self.total_w_loss3 = 0
        self.total_w_loss4 = 0
        self.train_sample_num = 0
        self.save_max = self.args.save_max
        self.logger.info("model path: %s " % self.model_path)

    def train(self,train, dev):
        self.logger.info("into train function model")
        with tf.Session(config=self.config) as sess:
            vs = tf.global_variables()
            self.logger.info("begin inital train model")
            sess.run(self.model.init_op)
            self.logger.info("over inital train model")
            if self.restore and((self.args.use_hvd and self.hvd.rank()==0) or (not self.args.use_hvd)):
                vs_parts=[]
                for v in vs:
                    if v.name.find('Adam') >= 0 or v.name.find('adam') >= 0: continue
                    if v.name.find('beta1_power') >= 0 or v.name.find('beta2_power') >= 0: continue
                    if v.name.find('beta1_power') >= 0 or v.name.find('beta2_power') >= 0: continue
                    if v.name.find('cnn') >= 0 : continue
                    vs_parts.append(v)
                self.logger.info("over load vsparts")
                #if (self.args.use_hvd and self.hvd.rank()==0) or (not self.args.use_hvd):  #intial parameter for the GPU 0
                saver = tf.train.Saver(vs_parts, max_to_keep=self.args.max_to_keep)
                saver.restore(sess, self.restore_model_path)
                sess.run(tf.assign(self.model_global_step, 0))
                saver = tf.train.Saver(vs, max_to_keep=self.args.max_to_keep)
                self.logger.info("over restore model")
            else:
                saver = tf.train.Saver(vs, max_to_keep=self.args.max_to_keep)
            if self.args.use_hvd: 
                self.logger.info("begin broadcase")
                sess.run(self.hvd.broadcast_global_variables(0)) #broadcast to other GPU
                self.logger.info("end broadcase")
            if not self.args.largefile:
                train_data=read_file(train,need_shuf=False,encoding=self.encoding,seq_mode=self.args.seq_mode)
            dev_paths = dev.split(",")
            all_dev_data={}
            for devpath in dev_paths:
                dev_data=[]
                for dev_batch in read_file_yeild(devpath,max_instances_size=self.batch_size,
                                       need_shuf=False,encoding=self.encoding,seq_mode=self.args.seq_mode):
                    dev_data.append(dev_batch)
                all_dev_data[devpath]=dev_data
            for epoch in range(self.epoch_num):
                if not self.args.largefile:
                    self.run_one_epoch(sess, train_data, all_dev_data, epoch, saver, max_saver=None)
                else:
                    for train_data in read_file_yeild(train,max_instances_size=1000000,
                                       need_shuf=False,encoding=self.encoding,seq_mode=self.args.seq_mode):
                        self.run_one_epoch(sess, train_data, all_dev_data, epoch, saver, max_saver=None)

    def extract(self,test,ofile):
        self.logger.info("into extract function model")
        saver = tf.train.Saver()
        if os.path.exists(ofile):
            os.remove(ofile)
        with tf.Session(config=self.config) as sess:
            saver.restore(sess,self.model_path)
            if not self.args.largefile:
                test_data=read_file(test,need_shuf=False,encoding=self.encoding,seq_mode=self.args.seq_mode)
                self.extract_ner(sess,test_data,ofile)
            else:
                for test_data in read_file_yeild(test,max_instances_size=1000000,
                                       need_shuf=False,encoding=self.encoding,seq_mode=self.args.seq_mode):
                    self.extract_ner(sess,test_data,ofile)
    
    def run_one_epoch(self, sess, train_data, all_dev_data, epoch, saver, max_saver=None):
        texts = []
        f_scores = [0.0]
        max_scores = 0.0
        self.local_step_num = 0
        self.total_w_count = 0
        self.total_w_loss = 0
        self.total_w_loss1 = 0
        self.total_w_loss2 = 0
        self.total_w_loss3 = 0
        self.total_w_loss4 = 0
        self.train_sample_num = 0
        self.start_time=datetime.now()

        def train_batch(texts):
            self.train_sample_num += len(texts)
            seqs, labels, segments, _, seqs_ori = bert_text_to_batch(texts, self.word2id, self.tag2id, self.dataset2flag)
            feed_dict, _, _, _ = self.model.get_feed_dict(seqs, labels, segments, self.lr, self.dropout)
            if self.global_step==-1: #print examples of batch globel_step
                for i in range(0,5):
                    sentence_ori=seqs_ori[i]
                    gold_tags=labels[i]
                    word_ids = seqs[i]
                    self.logger.info("".join(sentence_ori))
                    self.logger.info("_".join([self.id2tag[t] for t in gold_tags]))
                    self.logger.info("_".join([str(t) for t in word_ids]))
            real_w_count = sum(len(seq) for seq in seqs)
            self.total_w_count += real_w_count
            _, loss_train, self.global_step,loss1,loss2,loss3,loss4 = sess.run(
                [self.model.train_op, self.model.loss, self.model_global_step, 
                self.model.loss1,self.model.loss2,self.model.loss3,self.model.loss4], feed_dict=feed_dict)
            self.total_w_loss += loss_train * real_w_count
            self.total_w_loss1 += loss1 * real_w_count
            self.total_w_loss2 += loss2 * real_w_count
            self.total_w_loss3 += loss3 * real_w_count
            self.total_w_loss4 += loss4 * real_w_count
            if self.global_step > self.args.max_step:
                exit()
            if self.args.optimizer == "Adam":
                if self.global_step < self.args.warmup_step:
                    self.lr = self.args.lr * (1.0 * self.global_step / self.args.warmup_step)
                else:
                    self.lr = (1 - self.global_step * 1.0 / self.args.max_step) * (self.args.lr)
            else:
                pass
            if self.global_step % 1000 == 0:
                if (self.args.use_hvd and self.hvd.rank()==0) or (not self.args.use_hvd):
                    self.logger.info('{} seconds epoch {}, step {} , loss: {:.4}, loss1:{:.4}, loss2:{:.4}, loss3:{:.4} loss4:{:.4}'.format(
                    (datetime.now() - self.start_time).seconds, epoch, self.global_step, self.total_w_loss / self.total_w_count, self.total_w_loss1 / self.total_w_count, self.total_w_loss2 / self.total_w_count, self.total_w_loss3 / self.total_w_count, self.total_w_loss4 / self.total_w_count))
                    self.start_time=datetime.now()
            self.local_step_num += 1
            
        def eval_dev():
            self.logger.info('current train sample number: {}'.format(self.train_sample_num))
            self.logger.info('=========== eval dev ===========')
            word_f_socre = 0.0
            if not self.save_max:
                saver.save(sess, self.model_path, global_step=self.global_step)
            #for index, path in enumerate(dev_paths):
            index=0
            for path in all_dev_data:
                f_tag_score, f_word_socre = self.dev_ner(sess, all_dev_data[path], 32)
                self.logger.info("evaluate: %s" % path)
                self.logger.info("tag f-score: %.5f, word f-score: %.5f, last-best: %.5f" % (
                    f_tag_score, f_word_socre, self.max_scores))
                if index == 0:
                    f_scores.append(word_f_socre)
                    word_f_socre = f_word_socre
                self.logger.info("current lr: {}".format(self.lr))
                index+=1
            if word_f_socre >= self.max_scores:
                self.logger.info("found max dev score, save model...")
                self.max_scores = word_f_socre
                self.logger.info("current best f score: %.5f" % self.max_scores)
                if self.save_max :
                    saver.save(sess, self.model_path, global_step=self.global_step)
                    self.logger.info("save max f model success, global_step: {}".format(self.global_step))
                self.unseccessful_step_num = 0
            else:
                self.unseccessful_step_num += 1
                if self.unseccessful_step_num == 20:
                    if self.args.optimizer == "SGD":
                        #self.lr /= 2
                        if self.lr < 0.0001: self.lr = 0.0001  # the min lr
                    self.unseccessful_step_num = 0
                    self.logger.info("make new lr : {}".format(self.lr))
        
        def is_eval_step():
            if self.eval_step == 0:
                return False
            if self.global_step % self.eval_step == 0:
                return True
            return False
            
        #for instances_bag in read_file_yeild(train,max_instances_size=100000, need_shuf=False,encoding=self.encoding):
        instances_bag=train_data
        if self.args.use_hvd:
            mode_size = self.hvd.size()*self.batch_size - len(instances_bag)%(self.hvd.size()*self.batch_size)
            instances_bag.extend(instances_bag[0:mode_size])
            '''
            if len(instances_bag) > mode_size:
                instances_bag.extend(instances_bag[0:mode_size])
            elif len(instances_bag) > self.hvd.size():
                instances_bag.extend(instances_bag[0:self.hvd.size()])
            else:
                continue
            '''
            each_gpu_instance_sizes = int(len(instances_bag)/self.hvd.size())
            print(self.hvd.rank(),str(each_gpu_instance_sizes),self.hvd.size(),len(instances_bag),mode_size,self.batch_size)
            each_gpu_instances = instances_bag[int(self.hvd.rank())*each_gpu_instance_sizes:
                                           (int(self.hvd.rank())+1)*each_gpu_instance_sizes]
            self.logger.info("-----instances zone %s ------"%self.hvd.rank())
            self.logger.info("from %d to %d" %(int(self.hvd.rank())*each_gpu_instance_sizes,(int(self.hvd.rank())+1)*each_gpu_instance_sizes))
            instances=[]
            random.shuffle(instances)
            for instance in each_gpu_instances:
                instances.append(instance)
                if len(instances)==self.batch_size: 
                    train_batch(instances)
                    instances=[]
                    if is_eval_step() and self.hvd.rank()==0:
                        self.logger.info("*******")
                        eval_dev()
            if len(instances)!=0: 
                train_batch(instances)
        else:
            instances=[]
            for instance in instances_bag:
                instances.append(instance)
                if len(instances)==self.batch_size: 
                    train_batch(instances)
                    instances=[]
                    if is_eval_step():
                        self.logger.info("*******")
                        eval_dev()
            if len(instances)!=0: 
                train_batch(instances)

    def dev_ner(self, sess, dev_data, batch_size=64, do_print=False):
        token_ = [0] * 3
        men_ = [0] * 3
        begin_time = time.time()
        #self.logger.info("evaluate: %s" % dev)
        forward_duration = 0.
        sample_num = 0
        output=None
        Ner_eval = NerEvaluator(self.id2tag,encoding=self.encoding)
        y_trues,y_preds=[],[]
        for texts in dev_data:
        #for texts in read_file_yeild(dev, batch_size,need_shuf=True,encoding=self.encoding):
            if (sample_num % 32000 == 0 or sample_num % batch_size != 0 ) and do_print:
                self.logger.info("process line: %d" % sample_num)

            seqs, labels, segments, _, seqs_ori = bert_text_to_batch(texts, self.word2id, self.tag2id, self.dataset2flag)
            fwd_begin_time = time.time()
            label_array, seq_len_list, softmax, logits = self.predict_one_batch(sess, seqs, segments)
            fwd_end_time = time.time()
            forward_duration += fwd_end_time - fwd_begin_time
            assert len(seqs) == len(labels)
            assert len(seqs) == len(label_array)
            labels_ = process_label(labels)
            token_r, token_p, token_i = Ner_eval.evaluate(labels_, label_array, seqs, segments)
            men_r, men_p, men_i = Ner_eval.evaluate_mention(
                                      labels_, label_array, seqs_ori, softmax, texts, segments, output)
            # y_trues.extend(y_true)
            # y_preds.extend(y_pred)
            token_[0] += token_r
            token_[1] += token_p
            token_[2] += token_i
            men_[0] += men_r
            men_[1] += men_p
            men_[2] += men_i
        f_score = Ner_eval.metric("All_token", token_,self.logger)
        men_f_score = Ner_eval.metric("All_type", men_,self.logger)
        # self.logger.info(str(classification_report(y_trues, y_preds, digits=3)))
        #all_label = list(set(y_trues))
        return f_score,men_f_score

    def predict_one_batch(self, sess, seqs, segments, ngrams=None):
        feed_dict, seq_len_list, batch, max_len = self.model.get_feed_dict(seqs, None, segments, None, dropout=0, ngrams=ngrams)
        label_list, logits, softmax = sess.run(
        [self.model.labels_softmax_, self.model.logits, self.model.softmax],
        feed_dict=feed_dict)
        label_array = np.reshape(label_list, (batch, max_len))
        return label_array, seq_len_list, softmax, logits

    def extract_ner(self, sess, dev_data,resultfile,batch_size=64, do_print=False):

        def get_mention_bi(seqs, labels):
            '''
            seqs: word list
            labels: tag id list
            '''
            mention_input = []
            j = 0
            while j < len(seqs):
                '''
                if segs[j]==0:
                    j+=1
                    continue
                '''
                index_str = labels[j]
                if index_str == 1:
                    next = j + 1
                    while next < len(seqs):
                        next_str = labels[next]
                        if next_str == 2:
                            next += 1
                        else:
                            break
                    pos = "(%d,%d)" % (j, next)
                    mention = "".join(seqs[j:next])
                    mention_input.append(mention + pos)
                    j = next
                else:
                    j += 1
            return mention_input

        Ner_eval = NerEvaluator(self.id2tag,encoding=self.encoding)
        sample_num = 0
        texts=[]
        forward_duration = 0.0
        fw=open(resultfile,'w',encoding='utf-8')
        token_ = [0]*3
        men_ = [0]*3
        res = {}
        output = None
        for i in range(0,len(dev_data)):
            text=dev_data[i]
            if (sample_num % 32000 == 0 or sample_num % batch_size != 0 ):
                self.logger.info("process line: %d" % sample_num)
            texts.append(text)
            if len(texts)!= batch_size and i!=len(dev_data)-1:
                continue
            seqs, labels, segments, _, seqs_ori = bert_text_to_batch(texts, self.word2id, self.tag2id, self.dataset2flag)
            fwd_begin_time = time.time()
            label_array, seq_len_list, softmax, logits = self.predict_one_batch(sess, seqs, segments)
            token_r, token_p, token_i = Ner_eval.evaluate(labels, label_array, seqs, segments)
            men_r, men_p, men_i = Ner_eval.evaluate_mention(
                labels, label_array, seqs_ori, softmax, texts, segments, output)
            token_[0] += token_r
            token_[1] += token_p
            token_[2] += token_i
            men_[0] += men_r
            men_[1] += men_p
            men_[2] += men_i
            fwd_end_time = time.time()
            forward_duration += fwd_end_time - fwd_begin_time
            assert len(seqs) == len(labels)
            assert len(seqs) == len(label_array)
            assert len(seqs) == len(segments)
            for seq_idx, seq in enumerate(seqs):
                if len(seq) == 0: continue
                sentence = seq
                seq_len = seq_len_list[seq_idx]
                sentence_ori = seqs_ori[seq_idx]
                gold_tags = labels[seq_idx]
                pred_tags = label_array[seq_idx]
                seg_ori = segments[seq_idx]

                context_idx = seg_ori.index(1)
                entity_name = sentence_ori[1:context_idx-1] # [cls]
                sentence_ori = sentence_ori[context_idx:seq_len-1]
                gold_tags = gold_tags[context_idx:seq_len-1]
                pred_tags = pred_tags[context_idx:seq_len-1]
                mention_input = get_mention_bi(sentence_ori,gold_tags)
                mention_pred = get_mention_bi(sentence_ori,pred_tags)
                entity_name_str = "".join(entity_name)
                sentence_ori = "".join(sentence_ori)
                sentence_ori = sentence_ori.replace('[unused1]','')
                sentence_ori = sentence_ori.replace('[unused2]','') 
                if sentence_ori not in res:
                    res[sentence_ori] = {}
                    
                x = len(res[sentence_ori])
                res[sentence_ori][x+1] = {}
                res[sentence_ori][x+1]['name']=entity_name_str
                res[sentence_ori][x+1]['des']='|||'.join(mention_input)+'\t'+'|||'.join(mention_pred)
                #else:
                #    print(sentence_ori,entity_name_str)
            sample_num += len(texts)
            texts = []
        f_score = Ner_eval.metric("All_token", token_, self.logger)
        men_f_score = Ner_eval.metric("All_type", men_, self.logger)
        print(len(res)) 
        num=0
        for sent in res:
            fw.write(sent+'\n')
            num+=len(res[sent])
            for idx in res[sent]:
                fw.write(res[sent][idx]['name']+'\t')
                fw.write(res[sent][idx]['des']+'\n')
            fw.write('\n')
        print(num)
 

