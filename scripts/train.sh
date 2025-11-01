sbatch <<EOT
#!/bin/bash
#SBATCH --partition=      # Partition name
#SBATCH --gres=gpu:h200:2              # Number of GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export WANDB_PROJECT=GPTeacher

cd LLaMA-Factory
llamafactory-cli train ../configs/qwen3_0-6B_filter.yaml
EOT