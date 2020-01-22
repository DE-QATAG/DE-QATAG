#!/usr/bin/env python
import argparse
import os,sys,io,time,json
import tensorflow as tf
import horovod.tensorflow as hvd
from data import initial_embedding,set_result_path
from seq_model import SeqModel
from utils import str2bool
from utils import get_logger
from train_model import TrainFramework
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(),encoding='utf8')
## hyperparameters
parser = argparse.ArgumentParser(description='CNN-CRF for Chinese word segmentation task')
parser.add_argument('--encoding', type=str, default='utf-8', help='train data source')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='train data source')
parser.add_argument('--train_dat_path', type=str, default='./data_path/train.dat', help='train data source')
parser.add_argument('--valid_dat_path', type=str, default='./data_path/test.dat', help='train data source')
parser.add_argument('--test_dat_path', type=str, default='./data_path/test.dat', help='train data source')
parser.add_argument('--extract_result_path', type=str, default='./data_path/pred_result.dat', help='the extract resulf file')
parser.add_argument('--model_path', type=str, default='cnn', help='train data source')
parser.add_argument('--restore_model_path', type=str, default='cnn', help='train data source')
parser.add_argument('--word_to_id_path', type=str, required=True, help='word2id data source')
parser.add_argument('--tag_to_id_path', type=str, required=True, help='tag2id data source')
parser.add_argument('--dataset_to_flag_path', type=str, default=True, help='dataset2flag data source')
parser.add_argument('--bert_base_dir', type=str, default=True, help='dataset2flag data source')
parser.add_argument('--batch_size', type=int, default=32, help='#sample of each minibatch')
parser.add_argument('--layer_depth', type=int, default=6, help='#sample of each minibatch')
parser.add_argument('--filter_width', type=str, default='3', help='filter_width')
parser.add_argument('--num_filters', type=int, default=100, help='filter_width')
parser.add_argument('--epoch', type=int, default=10, help='#epoch of training')
parser.add_argument('--min_epoch', type=int, default=1, help='#min epoch of training')
parser.add_argument('--max_step', type=int, default=2000000000, help='epoch of training')
parser.add_argument('--eval_step', type=int, default=5000, help='evaluate every eval_step')
parser.add_argument('--warmup_step', type=int, default=2000, help='')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout keep_prob')
parser.add_argument('--mode', type=str, default='train', help='train/test/demo/dropout_pred')
parser.add_argument('--cuda_num', type=str, default='4', help='the folds need to produce result')
parser.add_argument('--hidden_dim', type=int, default=100, help='dim of hidden state')
parser.add_argument('--ngram_dim', type=int, default=100, help='dim of hidden state')
parser.add_argument('--embedding_dim', type=int, default=100, help='dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--restore', type=str2bool, default=False, help='')
parser.add_argument('--seq_mode', type=str, default='Ner', help='Ner or Seg')
parser.add_argument('--model_name', type=str, default='cnn_ngram', help='Bert or cnn_ngram')
parser.add_argument('--texts', type=str, default='conll2', help='ngram or conell noise ....')
parser.add_argument('--use_hvd', type=str2bool, default='True', help='hovod to speed')
parser.add_argument('--save_max', type=str2bool, default='True', help='.')
parser.add_argument('--max_to_keep', type=int, default='2', help='.')
parser.add_argument('--largefile', type=str2bool, default='True', help='read file or yeild read file')
parser.add_argument('--lambda1', type=float, default=1.0, help='tag loss')
parser.add_argument('--lambda2', type=float, default=1.0, help='description loss')
parser.add_argument('--lambda3', type=float, default=1.0, help='cls loss')
parser.add_argument('--lambda4', type=float, default=1.0, help='pair loss')





args = parser.parse_args()
paths=set_result_path(args.model_path)
logger = get_logger(paths['log_path'])
args_dict = vars(args)
args_str = json.dumps(args_dict, sort_keys=True, indent=4, separators=(',', ': '))
logger.info(args_str)

#Initialize Horovod and Session config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
if args.use_hvd:
    hvd.init()
    seed = hvd.rank()
    #logger.info("Hvd rank:",seed,",Local rank: ",hvd.local_rank(),",  hvd size :  ",hvd.size())
    config.gpu_options.visible_device_list = str(hvd.local_rank())
else:
    hvd=False
    #os.environ['CUDA_VISIBLE_DEVICES'] = 2,3
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# set embeddings
embeddings, bigram, trigram = None, None, None
if args.pretrain_embedding != 'random':
    embeddings, bigram, trigram = initial_embedding(args.embedding_dim,args.pretrain_embedding)
    logger.info("use pretrained embeddings")

## training model
if args.mode == 'train':
    if args.restore:
        restore_model_path = os.path.join(args.restore_model_path, "checkpoints")
        ckpt_file = tf.train.latest_checkpoint(restore_model_path)
        logger.info('restore_model_path: %s' % restore_model_path)
        paths['restore_model_path'] = ckpt_file
        logger.info('Restore from %s' % ckpt_file)
    model = SeqModel(args, logger,hvd)
    if args.model_name=='bert':
        model.build_bert_graph(args.bert_base_dir)
        print("Over build Bert model")
    elif args.model_name=='cnn_ngram':
        model.build_cnn_ngram_graph(args.num_filters,args.filter_width,embeddings,bigram,trigram)
    else:
        logger.info("Bad model name!!!!")
    logger.info("begin train:")
    seq_model_op = TrainFramework(model,paths,config)    
    seq_model_op.train(train=args.train_dat_path, dev=args.valid_dat_path)

## given the input data and output the predicted results
elif args.mode == 'extract':
    model_path = os.path.join(args.model_path, "checkpoints")
    ckpt_file = tf.train.latest_checkpoint(model_path)
    paths['model_path'] = ckpt_file
    logger.info('model_path: %s' % ckpt_file)
    model = SeqModel(args, logger)
    if args.model_name=='bert':
        model.build_bert_graph(args.bert_base_dir)
    elif args.model_name=='cnn_ngram':
        model.build_cnn_ngram_graph(args.num_filters,args.filter_width,embeddings,bigram,trigram)
    else:
        logger.info("Bad model name!!!!")
    test_path = args.test_dat_path
    logger.info("test_path: %s" % test_path)
    logger.info("begin test:")
    seq_model_op = TrainFramework(model,paths,config)    
    seq_model_op.extract(test=args.test_dat_path,ofile=args.extract_result_path)
else:
    logger.info('Error Mode parameter')
