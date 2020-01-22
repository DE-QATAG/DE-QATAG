#!/bin/bash
seqmode=Ner #Ner or Seg
#seqmode=Seg
encoding='utf-8'
dataset='per'
bert_base_dir="./conf/bert"
model_name=bert
if [[ "$seqmode" = "Ner" ]]; then
    data_dir='./data'
    train_dat_path="${data_dir}/${dataset}/pretrain.txt"
    valid_dat_path="${data_dir}/${dataset}/test.txt"
    test_dat_path="${data_dir}/${dataset}/test.txt"
    word_to_id_path="${bert_base_dir}/word2id.dat"
    tag_to_id_path="${data_dir}/tag2id.dat"
    dataset_to_flag_path="${data_dir}/dataset2flag.dat"
    extract_result_path="${data_dir}/${dataset}/res.txt"
elif [[ "$seqmode" = "Seg" ]]; then
    data_dir='./data/word_seg/utf8_version'
    train_dat_path="${data_dir}/${dataset}/bmes/train.txt"
    valid_dat_path="${data_dir}/${dataset}/bmes/test.txt"
    test_dat_path="${data_dir}/${dataset}/bmes/test.txt"
    #vocab="${data_dir}/${dataset}/raw/training_words.txt"
    word_to_id_path="${bert_base_dir}/word2id.dat"
    tag_to_id_path="${data_dir}/tag2id.dat"
    dataset_to_flag_path="${data_dir}/dataset2flag.dat"
else
    echo "bad seqmode!!!"
fi
model_dir='./model'
layer_depth=6
batch_size=8
epoch=40
max_step=6000000
dropout=0.1
lambda1=1.0
lambda2=1.0
lambda3=1.0
lambda4=0.1
optimizer=SGD
lr=0.2  # initial learning rate
restore='false' # true: used for finetuning,false: used for pretrain
use_hvd=True
eval_step=100
mode=train
# parameter setting
model_path=${model_dir}'/bert_'$dataset
if [[ "$restore" = "false" ]]; then
    model_path=${model_dir}'/bert_pretrain_'${dataset}
fi
restore_model_path=$model_path
echo $model_path
#mpirun --allow-run-as-root -np 4 -H localhost:4 python3 ./src/cnn_main.py \
#horovodrun -np 8 -H localhost:4,9.73.140.81:4  python3 ./src/cnn_main.py \
#horovodrun -np 1 -H localhost:1 
python3 ./seq2seq/run_main.py \
    --mode=${mode} \
    --word_to_id_path=${word_to_id_path} \
    --tag_to_id_path=${tag_to_id_path} \
    --dataset_to_flag_path=${dataset_to_flag_path} \
    --model_path=${model_path} \
    --restore_model_path=${restore_model_path} \
    --layer_depth=${layer_depth} \
    --dropout=${dropout} \
    --lr=${lr} \
    --optimizer=${optimizer} \
    --batch_size=${batch_size} \
    --epoch=${epoch} \
    --train_dat_path=${train_dat_path} \
    --valid_dat_path=${valid_dat_path} \
    --test_dat_path=${test_dat_path} \
    --extract_result_path=${extract_result_path} \
    --max_step=${max_step} \
    --eval_step=${eval_step} \
    --restore=${restore} \
    --use_hvd=${use_hvd} \
    --model_name=${model_name} \
    --seq_mode=${seqmode} \
    --encoding=${encoding} \
    --bert_base_dir=${bert_base_dir} \
    --lambda1=${lambda1} \
    --lambda2=${lambda2} \
    --lambda3=${lambda3} \
    --lambda4=${lambda4}
