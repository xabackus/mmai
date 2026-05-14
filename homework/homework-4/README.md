# Homework 4: GRPO for VLMs

## Overview

Reading assignment on Group Relative Policy Optimization (GRPO) via the DeepSeekMath paper, plus hands-on GRPO fine-tuning of a vision-language model.

## What I did

- **Reading assignment** on DeepSeekMath (Shao et al. 2024): analyzed how GRPO eliminates PPO's critic by using group-relative advantages, compared rule-based vs learned reward models and their respective failure modes, and worked through when GRPO is preferable to SFT (verifiable objectives, composite rewards, discovery of novel reasoning strategies).

- **Hands-on GRPO fine-tuning** of Kimi-VL on an A100 GPU. Implemented the full pipeline: group sampling, reward computation, advantage normalization, and policy gradient updates.

## Connection to the final project

This homework directly informed our understanding of alignment methods in EgoBlind-RA. The key insight from the reading — that GRPO maintains per-example supervision through group-relative advantages, unlike DPO which only provides pairwise rankings — explains one of our paper's central findings. In the final project, DPO failed on urgent queries because 1–3 word reference answers produce preference pairs with no meaningful variation. GRPO's group-based scoring would avoid this by comparing multiple completions per prompt rather than relying on a single chosen/rejected pair, which is why we cite it as the most promising future direction for the urgent regime (Section 7 of our paper). The reward hacking analysis from Question 2 also proved prescient: our DPO urgent adapter learned to minimize response length at the expense of accuracy, a form of reward hacking under the composite loss.

## Files

- `Xander_Backus_HW4_Reading_Assignment.pdf` — Written responses (3 questions on GRPO)
- `Homework_4_GRPO_VLMs.ipynb` — GRPO fine-tuning notebook
