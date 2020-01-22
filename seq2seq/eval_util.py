#!/usr/bin/env python
import numpy as np
import codecs
import ctypes
import os
import sys
python_version=3

class NerEvaluator:
    def __init__(self, id2tag,encoding='gbk'):
        self.id2tag= id2tag
        self.ENCODING=encoding 
    def tag_type(self, index, id2tag_):
        #tagname = id2tag_[index].encode(self.ENCODING).split("-")
        #print(type(id2tag_[index])) 
        #print(id2tag_[index])
        index_str = id2tag_[index]
        return index_str
    
    def metric(self,tokenname,token_,logger):
        try:
           p = token_[0] * 1.0 / token_[1]
           r = token_[0] * 1.0 / token_[2]
           f = p * r * 2 / (p + r)
           logger.info(("%s PRF result: %.4f\t%.4f\t%.4f   right,predict,total: %d, %d, %d ") % (
           tokenname,p, r,f, token_[0], token_[1], token_[2]))
        except:
           return 0
        return f
    def get_mention_bies(self, seqs, input, softmax, is_pred=False, masks=None):
        mention_input = []
        mention_dict = {}
        j = 1
        while j < len(seqs):
            if seqs[j] == 0:
                break
            if masks is not None and masks[j] == 0:
                j += 1
                continue
            # 4:S
            index = input[j]
            index_str, index_type = self.tag_type(index, self.id2tag)

            if index_str == 'S':
                pos = str(j)
                mention_input.append(seqs[j] + pos + index_type)
                mention_dict[pos] = (seqs[j], index_type)
                j += 1

            elif index_str == 'B':
                next = j + 1
                flag = False
                while next < len(seqs):
                    # 2:I
                    next_str, next_type = self.tag_type(input[next], self.id2tag)
                    if next_str == "I" and next_type == index_type:
                        next += 1
                        continue
                    # 3:E
                    elif next_str == "E" and next_type == index_type:
                        next += 1
                        flag = True
                        break
                    else:
                        break
                if flag:
                    pos = "(%d,%d)" % (j, next)
                    mention = "".join(seqs[j:next])
                    mention_input.append(mention + pos + index_type)
                    mention_dict[pos] = (mention, index_type)
                    j = next
                else:
                    j += 1
            else:
                j += 1
        return mention_input, mention_dict

    def get_mention_bi(self, seqs, labels, segments, softmax, is_pred=False, masks=None):
        '''
        seqs: word list
        labels: tag id list
        '''
        mention_input = []
        mention_dict = {}
        j = 1
        while j < len(seqs):
            if seqs[j] == 0:
                break
            if segments[j] == 0:    # question
                j += 1
                continue
            if masks is not None and masks[j] == 0:
                j += 1
                continue
            index_str = self.tag_type(labels[j], self.id2tag)
            if index_str == 'B-DESC':
                next = j + 1
                flag = False
                while next < len(seqs):
                    next_str = self.tag_type(labels[next], self.id2tag)
                    if next_str == "I-DESC":
                        next += 1
                    else:
                        break
                pos = "(%d,%d)" % (j, next)
                mention = "".join(seqs[j:next])
                mention_input.append(mention + pos)
                mention_dict[pos] = (mention)
                j = next
            else:
                j += 1
        
        return mention_input, mention_dict

    def evaluate_mention(self, label_input, label_pred, seqs, softmax, batch, segments, output, tag_scheme="bi",do_print=False, masks=None):
        '''
        label_input: label id list
        label_pred: preidct label id list
        seqs: words list
        '''
        assert len(label_input) == len(label_pred)
        assert len(label_input) >= len(seqs)
        r, p_t, l_t = 0, 0, 0
        for i in range(len(seqs)):
            seqs_code = seqs[i] # [str(s.encode(self.ENCODING)) for s in seqs[i]]
            if tag_scheme=='bie':
                #print("tagging scheme is bie")
                mention_input, true_dict = self.get_mention_bies(seqs_code, label_input[i], softmax[i], is_pred=False)
                mention_pred, pred_dict = self.get_mention_bies(seqs_code, label_pred[i], softmax[i], is_pred=True)
            elif tag_scheme=='bi':
                #print("tagging scheme is bi")
                mention_input, true_dict = self.get_mention_bi(seqs_code, label_input[i], segments[i], softmax[i], is_pred=False)
                mention_pred, pred_dict = self.get_mention_bi(seqs_code, label_pred[i], segments[i], softmax[i], is_pred=True)
            # pos_list = list(set(true_dict.keys()).union(set(pred_dict.keys())))
            flag = False
            print(mention_input)
            print(mention_pred)
            for men in mention_pred:
                if men in mention_input:
                    r += 1
                else:
                    flag = True
            for men in mention_input:
                if men not in mention_pred:
                    flag = True
            if not flag:
                if do_print and (not output is None):
                    output.write(batch[i].encode(self.ENCODING))
                    output.write("\n")
            # if flag:
            if True:
                if do_print:
                    print("***********")
                    chars = []
                    for index, char in enumerate(seqs_code):
                        label_id = label_input[i][index]
                        pred_id = label_pred[i][index]
                        if label_id != pred_id:
                            label = self.id2tag[label_id]
                            predict = self.id2tag[pred_id]
                            prob = str(softmax[i][index][pred_id])[0:5]
                            char = "%s[%s|%s:%s]" % (char, label.encode(self.ENCODING), predict.encode(self.ENCODING), prob)
                        chars.append(char)
                    print((" ".join(chars)).strip())
                    print("\t".join(mention_input))
                    print("\t".join(mention_pred))
            p_t += len(mention_pred)
            l_t += len(mention_input)
        return r, p_t, l_t

    # evaluate(token level)
    def evaluate(self, label_input, label_pred, seqs, segments, masks=None):
        assert len(label_input) == len(label_pred)
        assert len(label_input) == len(seqs)
        assert len(label_input) == len(segments)
        r = 0
        l_t = 0
        p_t = 0
        for i in range(len(label_input)):
            for j in range(len(label_input[i])):
                if seqs[i][j] == 0:
                    break
                if segments[i][j] == 0:
                    continue
                if masks is not None and masks[i][j] == 0:
                    continue
                if label_pred[i][j] != 0:
                    p_t += 1
                if label_input[i][j] != 0:
                    l_t += 1
                if label_input[i][j] == label_pred[i][j] and label_input[i][j] != 0:
                    r += 1
        return r, p_t, l_t
