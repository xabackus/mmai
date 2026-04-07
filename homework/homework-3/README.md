# Homework 3: Multimodal LLMs

## Overview
This assignment covers vision-language models end-to-end: baseline inference, prompt engineering, LoRA fine-tuning, and post-training evaluation. All experiments use EgoBlind data with Qwen2.5-VL-3B-Instruct.

## What I did
- Ran **baseline inference** on held-out EgoBlind images and found the pretrained model was "overly safe to the point of being useless and unsafe" — giving long, hedging answers that never actually told a blind user what to do. This failure mode directly motivated the urgency-aware response policies in our final project.
- **Prompt engineering**: discovered that telling the model it's an assistant for blind users substantially improved output quality — it started giving spatial/directional information instead of just listing objects. Adding urgency context helped further but couldn't fully fix the model's tendency to hedge.
- **LoRA fine-tuning** on EgoBlind with Qwen2.5-VL-3B-Instruct. Experimented with training set size, epochs, learning rate, and gradient accumulation. Best model achieved ~75% loss reduction with only a few hundred examples.
- **Post-training evaluation**: fine-tuned models gave much more useful, concise answers. However, all models still failed on spatial reasoning (e.g., consistently saying "right" when the correct answer was "left"), revealing a persistent weakness in egocentric spatial understanding.

## Connection to the final project
This homework was essentially a dry run of the final project's SFT pipeline. The finding that LoRA fine-tuning dramatically improves response quality even with small datasets gave us confidence to pursue the bifurcated adapter approach (Approach 1) in EgoBlind-RA. The spatial reasoning failures observed here also appear in our final baseline evaluation, confirming that this is a systematic challenge rather than a model-specific quirk. The prompt engineering experiments here informed the system prompts used in both the baseline and Approach 2 (prompt-conditioned unified model).

## Files
- `Xander's_copy_of_Homework_3_Multimodal_LLMs (2).ipynb` — full notebook with code, model outputs, and written responses
