# Homework 4: GRPO for VLMs

## Reading Assignment
Three questions on [DeepSeekMath](https://arxiv.org/abs/2402.03300) (Shao et al. 2024):
1. How GRPO eliminates PPO's critic by using group-relative advantages
2. Rule-based vs learned rewards and how reward hacking manifests differently in each
3. When to prefer SFT vs GRPO — SFT for high-quality paired data, GRPO for verifiable objectives and composite rewards

## Hands-On: GRPO Fine-Tuning
Notebook fine-tuning Kimi-VL with GRPO on an NVIDIA A100-SXM4-40GB. Covers reward function design, group sampling, and comparison with SFT baselines.

## Files
- `Xander_Backus_HW4_Reading_Assignment.pdf` — Written responses
- `Homework_4_GRPO_VLMs.ipynb` — GRPO fine-tuning notebook
