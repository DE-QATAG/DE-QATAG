#!/usr/bin/env python
# -*- coding: gbk -*-

import sys, pickle, os, random, io
import re
import numpy as np
import codecs
import ctypes
sys.stdout=io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')
SELF_DIR=os.path.dirname(os.path.abspath(__file__))
SO = ctypes.cdll.LoadLibrary
LIB = SO(os.path.join(SELF_DIR, 'libhash.so'))
BUCKET = 1500000
python_version=3


def set_result_path(output_path):
    paths = {}
    #output_path = args.model_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    summary_path = os.path.join(output_path, "summaries")
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)    
    paths['summary_path'] = summary_path
    model_path = os.path.join(output_path, "checkpoints")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    ckpt_prefix = os.path.join(model_path, "model")
    paths['model_path'] = ckpt_prefix
    result_path = os.path.join(output_path, "results")
    paths['result_path'] = result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    log_path = os.path.join(result_path, "log.txt")
    paths['log_path'] = log_path

    return paths

def initial_embedding(embedding_dim,train_data):
    if embedding_dim == 20:
        embedding_path = os.path.join('.', train_data, 'embeding_20/gram_1.w2v.txt')
        bigram_path = os.path.join('.', train_data, 'embeding_20/gram_2.w2v.txt')
        trigram_path = os.path.join('.', train_data, 'embeding_20/gram_3.w2v.txt')
    else:
        embedding_path = os.path.join('.', train_data, 'embeding_100/gram_1.w2v.txt')
        bigram_path = os.path.join('.', train_data, 'embeding_100/gram_2.w2v.txt')
        trigram_path = os.path.join('.', train_data, 'embeding_100/gram_3.w2v.txt')
    if os.path.exists(embedding_path):
        logger.info("load %s" % embedding_path)
        embeddings = load_embedding(embedding_path, word2id, embedding_dim)
    if os.path.exists(bigram_path):
        logger.info("load %s" % bigram_path)
        bigram = load_ngram(bigram_path, BUCKET, embedding_dim)
    if os.path.exists(trigram_path):
        logger.info("load %s" % trigram_path)
        trigram = load_ngram(trigram_path, BUCKET, embedding_dim)
    return embeddings, bigram, trigram

def load_dict(path, encoding='gbk'):
    word2id = {}
    id2word = {}
    with open(path,encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if line == '': continue
            columns = line.split('\t')
            if python_version==2:
                word = columns[0].decode(encoding)
            else:    
                word = columns[0]
            if word in word2id: continue
            idx = int(columns[1]) if len(columns) > 1 else len(word2id)
            word2id[word] = idx
            id2word[idx] = word
    return word2id, id2word

def load_dict_ano(path, encoding='gbk'):
    word2id = {}
    id2word = {}
    c = 0
    with open(path,encoding=encoding) as f:
        for line in f:
            row = line.strip().split('\t')
            if len(row) < 2:
                continue
            if python_version==2:           
                word = row[0].decode(encoding)
                id = row[1].decode(encoding)
            else:
                word=row[0]
                id=row[1]
            word2id[word] = id
            id2word[id] = word
            c += 1
    return word2id, id2word

def load_embedding(path, word2id, embedding_dim, encoding='gbk'):
    embedding = np.random.uniform(-1, 1, (len(word2id), embedding_dim))
    embedding[0, :] = np.zeros(embedding_dim)  # <PAD>
    count = 0
    with codecs.open(path, encoding=encoding) as f:
        for line in f:
            row = line.strip().split(' ')
            if row[0] in word2id:
                key = word2id[row[0]]
                embedding[key, :] = np.array(row[1:])
                count += 1
            else:
                pass
                #print row[0].encode("gbk")
    print("init word embedding num: %d" % count)
    #embedding = normalize(embedding, norm = 'l2')
    return embedding

def load_ngram(path, bucket, embedding_dim, encoding='gbk'):

    assert bucket == BUCKET
    embedding = np.random.uniform(-1, 1, (bucket, embedding_dim))
    embedding[0, :] = np.zeros(embedding_dim)  # <PAD>
    with codecs.open(path, encoding=encoding) as f:
        for line in f:
            row = line.strip().split(' ')
            if len(row) < 5:
                continue
            assert len(row) == embedding_dim + 1
            if row[0].isdigit():
                key = int(row[0])
                embedding[key, :] = np.array(row[1:])
            else:
                print (row[0] + "not digit")
    #embedding = normalize(embedding, norm = 'l2')
    #print embedding[100]
    return embedding

def load_yt_ngram(path, bucket, embedding_dim):
    embedding = np.random.uniform(-1, 1, (bucket, embedding_dim))
    embedding[0, :] = np.zeros(embedding_dim)  # <PAD>
    count = 0
    emb_dict = {}
    with codecs.open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = line.strip().split(' ')
                if len(row) < 20:
                    continue
                key = row[0].replace("@$","")
                key_id = bigram_hash(key.encode("gbk"))
                if not emb_dict.has_key(key_id):
                    emb_dict[key_id] = []
                emb_dict[key_id].append(np.array(row[1:]).astype(np.float32))
                #if len(emb_dict) > 1000:
                #    break
            except:
                pass
    
    yt_output = open("./data_path/yt.bigram","w")
    print("init ngram num: %d" % len(emb_dict))
    for k,v in emb_dict.items():
        if len(v) == 0:
            continue
        else:
            embed = np.mean(v,0)
            embed = [str(i) for i in embed]
            yt_output.write(str(k) + " " + " ".join(embed))
            yt_output.write("\n")
    yt_output.close()

    print("init ngram num: %d" % len(emb_dict))
    return embedding



def bigram_hash(t, bucket=500000):
    return LIB.hash(t, bucket)

def ngram2id(tokens, bucket=500000,encoding='gbk'):
    before_bi_ids = []
    after_bi_ids = []
    before_tri_ids = []
    after_tri_ids = []
    for idx, t in enumerate(tokens):
        if python_version==3:
            beginstr="<S>".encode(encoding)
            endstr="<E>".encode(encoding)
        else:
            beginstr="<S>"
            endstr="<E>"
        if idx == 0:
            before_bi_ids.append(bigram_hash(beginstr + t, bucket))
            before_tri_ids.append(bigram_hash(beginstr + t, bucket))
        elif idx == 1:
            before_bi_ids.append(bigram_hash(tokens[idx - 1] + t, bucket))
            before_tri_ids.append(bigram_hash(beginstr + tokens[idx - 1] + t, bucket))
        else:
            before_bi_ids.append(bigram_hash(tokens[idx - 1] + t, bucket))
            before_tri_ids.append(bigram_hash(tokens[idx - 2] + tokens[idx - 1] + t, bucket))

        if idx == len(tokens) - 1:
            after_bi_ids.append(bigram_hash(t +endstr, bucket))
            after_tri_ids.append(bigram_hash(t +endstr, bucket))
        elif idx == len(tokens) - 2:
            after_bi_ids.append(bigram_hash(t + tokens[idx + 1], bucket))
            after_tri_ids.append(bigram_hash(t + tokens[idx + 1] + endstr, bucket))
        else:
            after_bi_ids.append(bigram_hash(t + tokens[idx + 1], bucket))
            after_tri_ids.append(bigram_hash(t + tokens[idx + 1] + tokens[idx + 2], bucket))
    return before_bi_ids, after_bi_ids, before_tri_ids, after_tri_ids


def text2batch(texts, word2id, tag2id, dataset2flag, encoding='gbk',bucket=500000, add_cls=False):
    '''
    @input texts: [[line1,line2,linen],[line1,line2,linen]...]
                  line1 = word\ttag1\ttag2\t...
    @output:
       token_list_batch=[[wordid1,wordid2,...],[wordid1,wordid2]...]
       tag_list_batch=[[tagid1,tagid2,...],[tagid1,tagid2]...]

       datas_batch=[token_list_batch,tag_list_batch],
       ngrams=[ngramsinfo],
       raw_token_list_batch=[[word1,word2,...],[word1,word2]...]
    '''

    raw_token_list_batch = []
    token_list_batch, tag_list_batch, datas, ngrams = [], [], [], []
    before_bigrams, after_bigrams = [], []
    before_trigrams, after_trigrams = [], []
    for line in texts:
        raw_token_list = []
        tag_list = []
        for item in line:
            word_tag = item.split('\t')
            if len(word_tag) > 1:
                word = dataset2flag.get(word_tag[0], word_tag[0])
                raw_token_list.append(word)
                tag = tag2id[word_tag[-1]]
                tag_list.append(tag)
            elif len(word_tag) == 1:
                word = dataset2flag.get(word_tag[0], word_tag[0])
                raw_token_list.append(word)
                tag_list.append(0)
        assert len(raw_token_list) == len(tag_list)
        before_bi, after_bi, before_tri, after_tri = ngram2id([token.encode(encoding) for token in raw_token_list], bucket,encoding=encoding)
        before_bigrams.append(before_bi)
        after_bigrams.append(after_bi)
        before_trigrams.append(before_tri)
        after_trigrams.append(after_tri)
        raw_token_list_batch.append(raw_token_list)
        
        token_list = sentence2id(raw_token_list, word2id)
        token_list_batch.append(token_list)
        tag_list_batch.append(tag_list)
        datas.append((token_list, tag_list))
    ngrams = [before_bigrams, after_bigrams, before_trigrams, after_trigrams]
    return token_list_batch, tag_list_batch, datas, ngrams, raw_token_list_batch

def bert_text_to_batch(texts, word2id, tag2id, dataset2flag):
    '''
    @input texts: [[line1,line2,linen],[line1,line2,linen]...]
                  line1 = word\ttag1\ttag2\t...true tag
    @output:
       token_list_batch=[[wordid1,wordid2,...],[wordid1,wordid2]...]
       tag_list_batch=[[tagid1,tagid2,...],[tagid1,tagid2]...]

       datas_batch=[token_list_batch,tag_list_batch],
       raw_token_list_batch=[[word1,word2,...],[word1,word2]...]
    '''
    raw_token_list_batch = []
    token_list_batch, tag_list_batch, datas = [], [], []
    segment_list_batch = []
    for line in texts:
        raw_token_list = []
        tag_list = []
        segment_list = []
        for item in line:
            word_tag = item.split('\t')
            if len(word_tag) == 3:
                word = word_tag[0]
                raw_token_list.append(word)
                tag = tag2id[word_tag[1]]
                tag_list.append(tag)
                segment_list.append(int(word_tag[2]))
        assert len(raw_token_list) == len(tag_list)
        assert len(tag_list) == len(segment_list)
        raw_token_list_batch.append(raw_token_list)
        token_list = sentence2id(raw_token_list, word2id)
        token_list_batch.append(token_list)
        tag_list_batch.append(tag_list)
        segment_list_batch.append(segment_list)
        datas.append((token_list, tag_list, segment_list))
    return token_list_batch, tag_list_batch, segment_list_batch, datas, raw_token_list_batch


def read_corpus(corpus_path, word2id):
    data = []
    with codecs.open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    texts = []
    for line in lines:
        texts.append(line.strip())
    _, _, data = text2batch(texts, word2id)
    return data


def vocab_build(vocab_path, corpus_path, min_count, encoding='gbk'):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    word2id = {}
    with codecs.open(corpus_path, encoding=encoding) as fr:
        for line in fr:
            lines = line.strip().split("\n")
            words = lines[0].split(" ")
            for word in words:
                if word.isdigit():
                    word = '<NUM>'
                if word not in word2id:
                    word2id[word] = [len(word2id) + 1, 1]
                else:
                    word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word not in word2id:
            word = '[UNK]'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat

def process_label(label_pattern):
    label = []
    for row in label_pattern:
        tmpl = []
        for item in row:
            if item == 3 or item == 4:
                tmpl.append(0)
            else:
                tmpl.append(item)
        label.append(tmpl)
    return label


def process_label_pattern(label_pattern):
    label = []
    pattern = []
    cls = []
    for row in label_pattern:
        tmpl = []
        tmpp = []
        flag = False
        for item in row:
            if item == 3 or item == 4:
                tmpl.append(0)
                tmpp.append(1)
            else:
                if item == 1:
                    flag = True
                tmpl.append(item)
                tmpp.append(0)
        if flag:
            cls.append(1)
        else:
            cls.append(0)
        label.append(tmpl)
        pattern.append(tmpp)

    assert len(label) == len(label_pattern)
    assert len(pattern) == len(label)
    assert len(label) == len(cls)
    label_, _ = pad_sequences(label)
    label_pattern_, _ = pad_sequences(pattern, pad_mark=0)

    return label_, label_pattern_, cls

def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def cnn_pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list, masks = [], [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append([min(len(seq), max_len)])
        mask = [1] * len(seq) + [pad_mark] * (max_len - len(seq))
        masks.append(mask)
    return seq_list, seq_len_list, max_len, masks


def batch_yield(data, batch_size, vocab, tag2id, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2id:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, label_) in data:
        # sent_ = sentence2id(sent_, vocab)
        # label_ = [tag2id[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

def print_num_of_total_parameters(output_detail=False, output_to_logging=False):
	total_parameters = 0
	parameters_string = ""

	for variable in tf.trainable_variables():

		shape = variable.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
		total_parameters += variable_parameters
		if len(shape) == 1:
			parameters_string += ("%s %d, " % (variable.name, variable_parameters))
		else:
			parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))

	if output_to_logging:
		if output_detail:
			logging.info(parameters_string)
		logging.info("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
	else:
		if output_detail:
			print(parameters_string)
		print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))



def read_file(file_name, need_shuf=False, encoding='gbk',seq_mode="Ner"):
    '''
    @input file_name format: word \t tag1 \t tag2...\n \nword \t tag1 \t tag2...
    @output list:
    instances=[
       [line11,line2,line3,linen],
       [line21,line2,line3,linen],
       ...
       [linem1,line2,line3,linen],
    ]
    '''
    instances = []
    with open(file_name, encoding=encoding) as f:
        for line in f:
            if python_version == 2:
                line = line.strip().decode(encoding)
            else:
                line = line.strip()

            arr = line.split('\t')
            query = arr[0].split(' ')
            data = arr[1].split(' ')
            target = arr[2].split(' ')

            instance = ["[CLS]\tO\t0"]
            for item in query:
                instance.append(item+'\tO\t0')
            instance.append("[SEP]\tO\t0")
            for idx in range(len(data)):
                instance.append(data[idx]+'\t'+target[idx]+'\t1')
            instance.append("[SEP]\tO\t1")
            instances.append(instance)
    if need_shuf:
        random.shuffle(instances)
    return instances

def read_file_yeild(file_name, max_instances_size=100000, need_shuf=False, encoding='gbk',seq_mode="Ner"):
    '''
    @input file_name format: word \t tag1 \t tag2...\n \nword \t tag1 \t tag2...
    @output list:
    instances=[
       [line11,line2,line3,linen],
       [line21,line2,line3,linen],
       ...
       [linem1,line2,line3,linen],
    ]
    '''
    instances = []
    with open(file_name, encoding=encoding) as f:
        for line in f:
            if python_version == 2:
                line = line.strip().decode(encoding)
            else:
                line = line.strip()

            arr = line.split('\t')
            query = arr[0].split(' ')
            data = arr[1].split(' ')
            #
            # target = arr[2].replace('B-PER','O')
            # target = target.replace('I-PER','O')
            target = arr[2].split(' ')

            instance = ["[CLS]\tO\t0"]
            for item in query:
                instance.append(item + '\tO\t0')
            instance.append("[SEP]\tO\t0")
            
            idx = 0
            while idx < len(data):
                if target[idx] == 'B-PER':
                    instance.append('[unused1]\tB-PER\t1')
                    instance.append(data[idx]+'\tO\t1')
                    next = idx + 1
                    while next < len(data):
                        if target[next] == 'I-PER':
                            instance.append(data[next] + '\tO\t1')
                            next += 1
                        else:
                            break
                    instance.append('[unused2]\tI-PER\t1')
                    idx = next
                else:
                    instance.append(data[idx] + '\t' + target[idx] + '\t1')
                    idx += 1
            '''
            for idx in range(len(data)):
                instance.append(data[idx] + '\t' + target[idx] + '\t1') 
            '''
            instance.append("[SEP]\tO\t1")
            instances.append(instance)
            if len(instances) == max_instances_size:
                if need_shuf:
                    random.shuffle(instances)
                yield instances
                instances = []

    if len(instances) > 0:
        yield instances

def is_dataset_tag(word):
    return False
    return (int(word) >= 1 and int(word) <= 20)


if __name__ == "__main__":
    pass

