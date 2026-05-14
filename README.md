# MAS.S60 / 6.S985 — Multimodal AI (Spring 2026)

**Xander Backus** · MIT

Course portfolio for Multimodal AI, covering multimodal fusion, VLM fine-tuning, alignment, and agentic systems. Final project: [EgoBlind-RA](final_project/), a risk-adaptive egocentric visual assistance system for blind users.

## Repository

```
mmai/
├── homework/
│   ├── homework-1/   Data pipeline, EgoBlind preprocessing, metrics, prompt engineering
│   ├── homework-2/   Fusion & alignment: early/late/tensor/LMF, CLIP contrastive learning
│   ├── homework-3/   VLMs: baseline inference, prompt engineering, LoRA fine-tuning
│   ├── homework-4/   GRPO for VLMs: reading assignment + hands-on fine-tuning
│   └── homework-5/   Agentic AI: accessibility audit agent, safety eval, Discord deployment
└── final_project/    EgoBlind-RA: risk-adaptive routing for blind users
```

## Final Project: EgoBlind-RA

Joint project with [Julia Kim](https://github.com/juliavekim/mmai). We build a two-stage pipeline that classifies egocentric video queries from blind users as urgent or non-urgent, then routes each query to a response model calibrated to its safety stakes.

**Key results:**
- CLIP urgency classifier achieves 0.905 ROC-AUC on test
- Urgency-conditioned SFT reduces composite loss by 38% over the uniform-policy baseline
- DPO's effect depends on adapter architecture: hurts the unified adapter, helps the bifurcated non-urgent adapter
- The CLIP classifier closes 88% of the gap between zero-shot and oracle routing in the bifurcated pipeline

**Links:**
- [Joint repo](https://github.com/juliavekim/EgoBlind-RA)
- [LoRA adapters (HuggingFace)](https://huggingface.co/xabackus/egoblind-ra-adapters)
- [CLIP classifier (HuggingFace)](https://huggingface.co/julia225/egoblind-ra-clip-urgency)
- [Paper](final_project/paper/MMAI_Final_Report.pdf)
