# Homework 2: Multimodal Fusion and Alignment

## Overview
This assignment dives into the core techniques of multimodal learning: tensor operations, unimodal baselines, fusion architectures, and contrastive alignment (CLIP). All experiments are run on the EgoBlind dataset for binary urgency classification.

## What I did
- Built **unimodal baselines** for audio and image modalities on AV-MNIST, tuning hyperparameters (BatchNorm, dropout, learning rate) to understand single-modality ceilings.
- Implemented four fusion techniques **from scratch** on EgoBlind, fusing TF-IDF text features (128-dim) with video features (brightness, color, blur, edge density, optical flow):
  - **Early fusion** — concatenate before encoding
  - **Late fusion** — separate encoders, merge predictions
  - **Tensor fusion** — outer product of representations
  - **Low-Rank Tensor (LMF) fusion** — efficient approximation of tensor fusion
- Compared parameter counts, memory usage, and convergence time across all methods. Found that the larger multimodal models (tensor, LMF) overfit heavily on our relatively small dataset.
- Trained a **contrastive learning** model (CLIP-style) on EgoBlind frame-question pairs. Results were poor with extracted features alone, but the exercise directly motivated our final project's CLIP-based urgency classifier.

## Connection to the final project
The fusion experiments here taught us that naive multimodal fusion on EgoBlind overfits quickly — which is why we opted for **frozen CLIP features + lightweight MLP** in the final project's urgency classifier instead of end-to-end training. The contrastive learning section was essentially a prototype of what became Julia's CLIP urgency classifier (0.905 ROC-AUC).

## Files
- `Xander's_Final_Copy_of_Homework_2_Multimodal_Fusion_and_Alignment.ipynb` — full notebook with code, visualizations, and written responses
