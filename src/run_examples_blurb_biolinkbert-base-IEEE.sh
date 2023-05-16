#!/bin/bash

#$ -q gpu@@lalor           # Specify queue
#$ -l gpu_card=2
#$ -pe smp 1
#$ -N LinkDblurbIEEE       # Specify job name
#$ -t 3-4

module load cuda      # Required modules
module load cudnn
source activate linkbert

echo $SGE_TASK_ID


#export MODEL=BioLinkBERT-base
#export MODEL_PATH=michiyasunaga/$MODEL
#export MODEL=BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
#export MODEL_PATH=microsoft/$MODEL
#export MODEL=bert-base-uncased
#export MODEL_PATH=$MODEL
export MODEL=biobert-v1.1
export MODEL_PATH=dmis-lab/$MODEL


############################### QA: PubMedQA ###############################
if [[ $SGE_TASK_ID -eq 1 ]]; then
baseline=baseline
task=pubmedqa_hf
datadir=~/data/linkbert/seqcls/$task
outdir=runs/$baseline/$task/$MODEL
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
baseline=baseline
task=bioasq_hf
datadir=~/data/linkbert/seqcls/$task
outdir=runs/$baseline/$task/$MODEL
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
baseline=difflength
task=pubmedqa_hf
datadir=~/data/linkbert/seqcls/$task
outdir=runs/$baseline/$task/$MODEL
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
baseline=difflength
task=bioasq_hf
datadir=~/data/linkbert/seqcls/$task
outdir=runs/$baseline/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 2e-5 --warmup_steps 100 --num_train_epochs 20 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  --baseline $baseline \
  |& tee $outdir/log.txt 

########### DIFFPERP #############################
############################### QA: PubMedQA ###############################
elif [[ $SGE_TASK_ID -eq 5 ]]; then
baseline=diffperp
task=pubmedqa_hf
datadir=~/data/linkbert/seqcls/$task
outdir=runs/$baseline/$task/$MODEL
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
elif [[ $SGE_TASK_ID -eq 6 ]]; then
baseline=diffperp
task=bioasq_hf
datadir=~/data/linkbert/seqcls/$task
outdir=runs/$baseline/$task/$MODEL
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
