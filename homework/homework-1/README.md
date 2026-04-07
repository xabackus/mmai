# Homework 1: Datasets and Preprocessing

## Overview
This assignment covers the data pipeline foundations for multimodal learning: extracting and aligning modalities from raw data, visualizing distributions, choosing evaluation metrics, and experimenting with instruction tuning.

## What I did
- Preprocessed the **EgoBlind** dataset — the same dataset that became the foundation of our final project. Dealt with the zero-padding mismatch between `video_name` in the CSV (plain integers) and the actual filenames (`00000.mp4`, `00001.mp4`, ...), a bug that resurfaced multiple times later in the project.
- Extracted video frames and visualized the data distribution, sample frames, and input embeddings via t-SNE.
- Defined evaluation metrics for both components of our planned system: **F1** for the urgency classifier (because the class balance is skewed toward non-urgent) and a **composite loss** for the response generator that balances accuracy, utility, and latency.
- Experimented with instruction tuning / prompt engineering for sentiment classification and visual QA tasks.

## Connection to the final project
This is where the EgoBlind-RA project started. The dataset exploration here — understanding the question types, the video quality challenges, and the urgency distribution — directly informed our decision to build a risk-adaptive routing system. The zero-padding bug discovered here saved us hours of debugging later.

## Files
- `Xander's_copy_of_mmai_HW1.ipynb` - full notebook with code and written responses
