#!/bin/bash

#$ -M jlalor1@nd.edu   # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -q gpu@@lalor           # Specify queue
#$ -l gpu_card=2
#$ -pe smp 1
#$ -N LinkDmedqa       # Specify job name
#$ -t 1-12          

module load cuda      # Required modules
module load cudnn
source activate linkbert

#SGE_TASK_ID=1
echo $SGE_TASK_ID


if [[ $(($SGE_TASK_ID%4)) -eq 0 ]]; then
export MODEL=BioLinkBERT-base
export MODEL_PATH=michiyasunaga/$MODEL
elif [[ $(($SGE_TASK_ID%4)) -eq 1 ]]; then
export MODEL=bert-base-uncased
export MODEL_PATH=$MODEL
elif [[ $(($SGE_TASK_ID%4)) -eq 2 ]]; then
export MODEL=biobert-v1.1
export MODEL_PATH=dmis-lab/$MODEL
elif [[ $(($SGE_TASK_ID%4)) -eq 3 ]]; then
export MODEL=biomed_roberta_base
export MODEL_PATH=allenai/$MODEL
fi
echo $MODEL_PATH
############################### MedQA ###############################

if [[ $(($SGE_TASK_ID%3)) -eq 0 ]]; then
echo "test1"
task=medqa_usmle_hf
datadir=~/data/linkbert/mc/$task
outdir=runs/baseline/$task/$MODEL
mkdir -p $outdir
python3 -u mc/run_multiple_choice.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 --warmup_steps 100 --num_train_epochs 2 --max_seq_length 512 --fp16 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  --baseline baseline \
  |& tee $outdir/log.txt 

elif [[ $(($SGE_TASK_ID%3)) -eq 1 ]]; then
echo "test2"
task=medqa_usmle_hf
datadir=~/data/linkbert/mc/$task
outdir=runs/difflength/$task/$MODEL
mkdir -p $outdir
python3 -u mc/run_multiple_choice.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 --warmup_steps 100 --num_train_epochs 2 --max_seq_length 512 --fp16 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  --baseline difflength \
  |& tee $outdir/log.txt 
elif [[ $(($SGE_TASK_ID%3)) -eq 2 ]]; then
echo "test3"
task=medqa_usmle_hf
datadir=~/data/linkbert/mc/$task
outdir=runs/diffperp/$task/$MODEL
mkdir -p $outdir
python3 -u mc/run_multiple_choice.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 --warmup_steps 100 --num_train_epochs 2 --max_seq_length 512 --fp16 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  --baseline diffperp --local_rank -1 \
  |& tee $outdir/log.txt 

fi
