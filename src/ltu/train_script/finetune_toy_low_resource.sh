#!/bin/bash
#SBATCH -o ./log/%j_alm.out
#SBATCH --error ./log/%j_alm.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=A40short
#SBATCH --ntasks-per-node=32

source ${HOME}/.bashrc
# use model parallel if you have multiple small gpus on a single node, will be slower
# tune micro_batch_size to be the largest value that does not cause OOM

export TRANSFORMERS_CACHE=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/
output_dir='../exp/ltu_ft_toy_low_resource_full/'
mkdir -p $output_dir
cp "$0" ${output_dir}/$(date +"%Y-%m-%d-%H-%M-%S").sh

python ../finetune_low_resource.py \
    --base_model '/home/s6kogase/seminar/trained/full_ft_2e-5_20000.bin' \
    --data_path '/home/s6kogase/seminar/ltu/openaqa/data/deepfakes/audio_classification_data.json' \
    --output_dir $output_dir \
    --batch_size 256 \
    --micro_batch_size 1 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --cutoff_len 100 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --wandb_run_name ${output_dir} \
    --save_steps 10 \
    --trainable_params all