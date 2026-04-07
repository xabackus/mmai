# MMAI 2026 — Xander Backus

My portfolio for Modeling Multimodal AI (MAS.S60 / 6.S985), Spring 2026 at MIT.

## About me

I'm an MIT undergrad pursuing a BS in AI and Decision Making + Mathematics (with a Physics minor) alongside an MEng in EECS. My research has spanned machine learning for acoustic holograms at CSAIL, renewable energy optimization at LIDS, and now dual-timescale memory systems for video agents. I am interested in all things related to Large Multimodal Models.

## How this repo is organized

Each homework builds toward the final project.

```
mmai/
├── homework/
│   ├── homework-1/    # Data pipeline: EgoBlind preprocessing, metrics, prompt engineering
│   ├── homework-2/    # Fusion & alignment: early/late/tensor/LMF fusion, CLIP contrastive learning
│   └── homework-3/    # VLMs: baseline inference, prompt engineering, LoRA fine-tuning on EgoBlind
├── final_project/     # EgoBlind-RA: risk-adaptive routing for blind users
└── README.md
```

## Final Project: EgoBlind-RA

Built with Julia Kim. A risk-adaptive framework that classifies blind users' egocentric video queries by urgency and routes them to specialized response policies.

→ [Shared project repo](https://github.com/juliavekim/EgoBlind-RA) · [Julia's portfolio](https://github.com/juliavekim/mmai)

**My contributions:** urgency annotation pipeline (GPT-5.2), Kimi API baseline evaluation (4,029 examples), composite loss function design, hyperparameter grid search (3,888 configs), LoRA SFT on MIT Engaging cluster, evaluation pipeline.

## Homework

| # | Topic | Key takeaway for final project |
|---|-------|-------------------------------|
| [HW1](homework/homework-1/) | Datasets & preprocessing | Discovered EgoBlind's zero-padding quirks; defined F1 + composite loss metrics |
| [HW2](homework/homework-2/) | Fusion & contrastive learning | Learned that heavy fusion overfits on small data → motivated frozen CLIP + MLP classifier |
| [HW3](homework/homework-3/) | VLMs & LoRA fine-tuning | Confirmed LoRA works well even with few examples; found spatial reasoning failures persist across models |
