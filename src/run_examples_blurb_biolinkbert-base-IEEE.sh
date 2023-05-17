#!/bin/bash

#$ -q gpu@@lalor           # Specify queue
#$ -l gpu_card=2
#$ -pe smp 1
#$ -N LinkDblurbIEEE       # Specify job name
#$ -t 1-4

module load cuda      # Required modules
module load cudnn
source activate linkbert

echo $SGE_TASK_ID

#task=bioasq_hf
task=pubmedqa_hf


############################### QA: PubMedQA ###############################
if [[ $SGE_TASK_ID -eq 1 ]]; then
export MODEL=BioLinkBERT-base
export MODEL_PATH=michiyasunaga/$MODEL
baseline=fullmodel
datadir=~/data/linkbert/seqcls/$task
outdir=runs/selfguided/$baseline/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 2e-5 --warmup_steps 100 --num_train_epochs 30 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  --baseline $baseline \
  |& tee $outdir/log.txt 

############################### QA: BioASQ ###############################
elif [[ $SGE_TASK_ID -eq 2 ]]; then
export MODEL=BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
export MODEL_PATH=microsoft/$MODEL
baseline=fullmodel
datadir=~/data/linkbert/seqcls/$task
outdir=runs/selfguided/$baseline/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 2e-5 --warmup_steps 100 --num_train_epochs 20 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  --baseline $baseline \
  |& tee $outdir/log.txt 

############## DIFFLENGTH #########################
############################### QA: PubMedQA ###############################
elif [[ $SGE_TASK_ID -eq 3 ]]; then
export MODEL=bert-base-uncased
export MODEL_PATH=$MODEL
baseline=fullmodel
datadir=~/data/linkbert/seqcls/$task
outdir=runs/selfguided/$baseline/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 2e-5 --warmup_steps 100 --num_train_epochs 30 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  --baseline $baseline \
  |& tee $outdir/log.txt 

############################### QA: BioASQ ###############################
elif [[ $SGE_TASK_ID -eq 4 ]]; then
export MODEL=biobert-v1.1
export MODEL_PATH=dmis-lab/$MODEL
baseline=fullmodel
datadir=~/data/linkbert/seqcls/$task
outdir=runs/selfguided/$baseline/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 2e-5 --warmup_steps 100 --num_train_epochs 20 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  --baseline $baseline \
  |& tee $outdir/log.txt 

fi
