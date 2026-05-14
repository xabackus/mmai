#!/bin/bash
#SBATCH --job-name=eval-clip-pipe
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:50:00
#SBATCH --output=logs/eval_pipeline_clip_%j.out
#SBATCH --error=logs/eval_pipeline_clip_%j.err

mkdir -p ~/logs
source /etc/profile
module load miniforge
conda activate egoblind-eval

pip install git+https://github.com/openai/CLIP.git 2>&1

echo "Starting CLIP pipeline evaluation at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo ""

python ~/evaluate_pipeline_clip.py

echo ""
echo "Finished at $(date)"
