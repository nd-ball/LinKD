#!/bin/bash

#$ -q gpu@@lalor           # Specify queue
#$ -l gpu_card=4
#$ -pe smp 1
#$ -N LinkDmedqa       # Specify job name
#$ -t 1-2              

module load cuda      # Required modules
module load cudnn
source activate linkbert

echo $SGE_TASK_ID


export MODEL=BioLinkBERT-base
export MODEL_PATH=michiyasunaga/$MODEL

############################### MedQA ###############################

if [[ $SGE_TASK_ID -eq 1 ]]; then

task=medqa_usmle_hf
datadir=~/data/linkbert/mc/$task
outdir=runs/baseline/$task/$MODEL
mkdir -p $outdir
python3 -u mc/run_multiple_choice.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 --warmup_steps 100 --num_train_epochs 2 --max_seq_length 512 --fp16 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  --baseline True \
  |& tee $outdir/log.txt &

elif [[ $SGE_TASK_ID -eq 2 ]]; then

task=medqa_usmle_hf
datadir=~/data/linkbert/mc/$task
outdir=runs/difflength/$task/$MODEL
mkdir -p $outdir
python3 -u mc/run_multiple_choice.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 --warmup_steps 100 --num_train_epochs 2 --max_seq_length 512 --fp16 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &

fi