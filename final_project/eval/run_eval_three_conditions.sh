#!/bin/bash
#SBATCH --job-name=eval-3cond
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:50:00
#SBATCH --output=logs/eval_three_conditions_%j.out
#SBATCH --error=logs/eval_three_conditions_%j.err

mkdir -p ~/logs
source /etc/profile
module load miniforge
conda activate egoblind-eval

echo "Starting three-condition evaluation at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo ""
echo "Condition 1: Base model (no fine-tuning, no routing)"
echo "Condition 2: Oracle routing (ground-truth urgency -> best models)"
echo "Condition 3: Classifier pipeline (base model classifies -> routes)"
echo ""

python ~/evaluate_three_conditions.py

echo ""
echo "Finished at $(date)"
