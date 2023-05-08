#!/bin/bash

#$ -q gpu@@lalor           # Specify queue
#$ -l gpu_card=1
#$ -pe smp 1
#$ -N LinkDblurb       # Specify job name
#$ -t 1-13

module load cuda      # Required modules
module load cudnn
source activate linkbert

echo $SGE_TASK_ID


export MODEL=BioLinkBERT-base
export MODEL_PATH=michiyasunaga/$MODEL

############################### QA: PubMedQA ###############################
if [[ $SGE_TASK_ID -eq 1 ]]; then
task=pubmedqa_hf
datadir=~/data/linkbert/seqcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 2e-5 --warmup_steps 100 --num_train_epochs 30 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt 

############################### QA: BioASQ ###############################
elif [[ $SGE_TASK_ID -eq 2 ]]; then
task=bioasq_hf
datadir=~/data/linkbert/seqcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 2e-5 --warmup_steps 100 --num_train_epochs 20 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt 



############################### BIOSSES ###############################
elif [[ $SGE_TASK_ID -eq 3 ]]; then
task=BIOSSES_hf
datadir=~/data/linkbert/seqcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --metric_name pearsonr \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 1e-5 --num_train_epochs 30 --max_seq_length 512 --seed 5 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt 



############################### HoC ###############################
elif [[ $SGE_TASK_ID -eq 4 ]]; then
task=HoC_hf
datadir=~/data/linkbert/seqcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --metric_name hoc \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 4e-5 --num_train_epochs 40 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt 



############################### RE: ChemProt ###############################
elif [[ $SGE_TASK_ID -eq 5 ]]; then
task=chemprot_hf
datadir=~/data/linkbert/seqcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --metric_name PRF1 \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 3e-5 --num_train_epochs 10 --max_seq_length 256 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt 

############################### RE: DDI ###############################
elif [[ $SGE_TASK_ID -eq 6 ]]; then
task=DDI_hf
datadir=~/data/linkbert/seqcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --metric_name PRF1 \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 2e-5 --num_train_epochs 5 --max_seq_length 256 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt 

############################### RE: GAD ###############################
elif [[ $SGE_TASK_ID -eq 7 ]]; then
task=GAD_hf
datadir=~/data/linkbert/seqcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --metric_name PRF1 \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 3e-5 --num_train_epochs 10 --max_seq_length 256 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt 



############################### EBM PICO ###############################
elif [[ $SGE_TASK_ID -eq 8 ]]; then
task=ebmnlp_hf
datadir=~/data/linkbert/tokcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u tokcls/run_ner.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --return_macro_metrics \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 5e-5 --num_train_epochs 1 --max_seq_length 512  \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt 



############################### NER: JNLPBA ###############################
elif [[ $SGE_TASK_ID -eq 9 ]]; then
task=JNLPBA_hf
datadir=~/data/linkbert/tokcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u tokcls/run_ner.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
   --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 1e-5 --warmup_ratio 0.1 --num_train_epochs 5 --max_seq_length 512  \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt 

############################### NER: NCBI-disease ###############################
elif [[ $SGE_TASK_ID -eq 10 ]]; then
task=NCBI-disease_hf
datadir=~/data/linkbert/tokcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u tokcls/run_ner.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 5e-5 --warmup_ratio 0.1 --num_train_epochs 20 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt 

############################### NER: BC2GM ###############################
elif [[ $SGE_TASK_ID -eq 11 ]]; then
task=BC2GM_hf
datadir=~/data/linkbert/tokcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u tokcls/run_ner.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 6e-5 --warmup_ratio 0.1 --num_train_epochs 50 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt 

############################### NER: BC5CDR-disease ###############################
elif [[ $SGE_TASK_ID -eq 12 ]]; then
task=BC5CDR-disease_hf
datadir=~/data/linkbert/tokcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u tokcls/run_ner.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 5e-5 --warmup_ratio 0.1 --num_train_epochs 8 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt 

############################### NER: BC5CDR-chem ###############################
elif [[ $SGE_TASK_ID -eq 13 ]]; then
task=BC5CDR-chem_hf
datadir=~/data/linkbert/tokcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u tokcls/run_ner.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 5e-5 --warmup_ratio 0.1 --num_train_epochs 20 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt 
fi