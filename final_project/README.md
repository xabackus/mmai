# EgoBlind-RA: Risk-Adaptive Egocentric Visual Assistance for Blind Users

**Xander Backus** and **Julia Kim** · MAS.S60 / 6.S985 · Spring 2026

Blind users wearing cameras ask questions about their surroundings. Some questions are urgent ("Is there a step in front of me?"), others are not ("What brand of cereal is this?"). Current VLMs treat both identically. EgoBlind-RA classifies query urgency from video + text, then routes to a response calibrated to the safety stakes.

## Architecture

```
Video + Question → CLIP Classifier (frozen ViT-B/32 + MLP) → urgency prediction
                                                                    │
                              ┌─────────────────────────────────────┴──────────────────────┐
                              ▼                                                            ▼
                     Approach 1 (Xander)                                         Approach 2 (Julia)
                     Separate LoRA adapters                                      Single LoRA adapter
                     urgent → SFT adapter                                        [URGENT]/[NON-URGENT]
                     non-urgent → SFT+DPO adapter                                prompt tag conditioning
                              │                                                            │
                              └─────────────────────────────────────┬──────────────────────┘
                                                                    ▼
                                                        Urgency-calibrated response
```

Base model: [Kimi-VL-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct)

## Results

| Routing | Approach 1 Loss | Approach 2 Loss |
|---------|----------------|----------------|
| Baseline (uniform) | 0.607 | 0.607 |
| CLIP classifier | 0.556 | 0.375 |
| Oracle (GT labels) | 0.549 | 0.375 |

**DPO finding:** DPO hurts when reference answers are short (1–3 words for urgent queries — no variation in preference pairs). DPO helps when references are longer (non-urgent). The bifurcated design isolates these regimes; the unified design mixes them. This is why DPO regresses Approach 2 but partially helps Approach 1.

**Robustness finding:** Approach 2's unified adapter matches oracle loss despite 21.5% of test-time urgency tags being wrong. Approach 1 degrades under misrouting. For deployment, Approach 2 is safer.

## Links

| Resource | URL |
|----------|-----|
| Joint repo | [juliavekim/EgoBlind-RA](https://github.com/juliavekim/EgoBlind-RA) |
| Approach 1 adapters | [xabackus/egoblind-ra-adapters](https://huggingface.co/xabackus/egoblind-ra-adapters) |
| CLIP classifier | [julia225/egoblind-ra-clip-urgency](https://huggingface.co/julia225/egoblind-ra-clip-urgency) |
| EgoBlind dataset | [Xiao et al. 2025](https://arxiv.org/abs/2503.08221) |

## Directory Structure

```
final_project/
├── paper/                     NeurIPS-format report
│   ├── MMAI_Final_Report.pdf
│   ├── MMAI_Final_Report.tex
│   ├── ref.bib
│   └── pipeline.pdf
├── presentation/              Beamer slides (presented May 12)
│   ├── EgoBlind_RA.pdf
│   └── EgoBlind_RA.tex
├── eval/                      Evaluation scripts (run on MIT Engaging, L40S GPUs)
│   ├── evaluate_three_conditions.py    3-condition eval (base, oracle, zero-shot pipeline)
│   ├── evaluate_pipeline_clip.py       Full pipeline with Julia's CLIP classifier
│   ├── rescore_pipeline_canonical.py   Re-score with paper's canonical loss params
│   ├── run_eval_three_conditions.sh    SLURM job script
│   └── run_eval_pipeline_clip.sh       SLURM job script
├── training/                  Training scripts and configs
│   ├── prepare_vision_sft.py           Build vision SFT datasets
│   ├── generate_dpo_pairs_vision*.py   DPO pair generation (4 variants)
│   └── configs/                        LLaMA-Factory YAML configs (6 adapters)
├── results/                   Summary JSONs
│   ├── eval_summary_pipeline_clip.json
│   └── FINAL_summary_three_conditions.json
└── ONBOARDING_DOCUMENT.md     Full project context for LLM-assisted development
```

## Training on MIT Engaging

SSH: `athena.dialup.mit.edu` → `orcd-login.mit.edu`
GPU: NVIDIA L40S (46GB), partition `mit_normal_gpu`
Conda: `egoblind` (training) / `egoblind-eval` (inference)

All adapters use LoRA rank 8, targeting q_proj + v_proj, 1 epoch, `kimi_vl_nothink` template. The urgent adapter trains at lr=1e-4, the non-urgent at lr=2e-5 (higher rate causes NaN on longer targets). Vision conditioning is required — text-only training collapses to "Yes"/"No" pattern matching.

## Citation

```bibtex
@misc{kim2026egoblindra,
  title  = {EgoBlind-RA: Towards Safer Egocentric Assistive AI for Blind Users via Risk-Adaptive Routing},
  author = {Kim, Julia and Backus, Xander},
  year   = {2026},
  url    = {https://github.com/juliavekim/EgoBlind-RA},
}
```
