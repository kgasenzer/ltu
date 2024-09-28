#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=eval
#SBATCH --gres=gpu:1
#SBATCH --partition A40short
#SBATCH --time=8:00:00
#SBATCH --output=./eval_res/eval_%j.out
#SBATCH --error=./eval_res/eval_%j.err
source ${HOME}/.bashrc
ml CUDA/11.7

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

echo "Got nodes:"
echo $SLURM_JOB_NODELIST
echo "Jobs per node:"
echo $SLURM_JOB_NUM_NODES

echo -e "Evaluating..."
python eval/eval_fsd50k.py