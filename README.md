# EgoBlind-RA

Risk-adaptive egocentric visual assistance for blind and low-vision users. Classifies queries by urgency and routes them to response policies matched to safety stakes. Evaluated on the [EgoBlind benchmark](https://arxiv.org/abs/2503.08221).

**Julia Kim & Xander Backus** · MIT · Equal contribution

## Overview

Existing multimodal models apply a single inference policy across all queries, ignoring variability in urgency and risk. EgoBlind-RA addresses this by classifying each query as *urgent* or *non-urgent* and routing it accordingly:

- **Urgent queries** (navigation hazards, safety-critical) → low-latency, conservative guidance
- **Non-urgent queries** → additional computation for higher-quality answers

We implement and compare two architectures:

| | Approach 1: Xander | Approach 2: Julia |
|-|---------------------|---------------------|
| Models | Two separate LoRA adapters | One single LoRA adapter |
| Routing | Classifier → different models | Classifier → different system prompts |
| SFT | Two training runs (urgent, non-urgent) | One training run (mixed, tagged) |
| DPO | On urgent model only | On unified model with `[URGENT]` tag |
| Inference VRAM | 2× model loads | 1× model load |

The core question: can a single model learn to behave differently based on an urgency tag, or does hard-routing to specialized models produce better results?

## Repo structure

```
EgoBlind-RA/
├── data/
│   ├── train_labeled.csv              # GPT-annotated urgency labels (train)
│   ├── test_labeled.csv               # GPT-annotated urgency labels (test)
│   └── baseline_frames/               # extracted video frames for baseline eval
├── docs/
│   ├── project_proposal.pdf
│   └── midterm_report.pdf
├── models/
│   └── CLIP_Urgency_Classifier.ipynb  # CLIP-based binary urgency classifier
├── scripts/
│   ├── classify_urgency.py            # GPT labeling pipeline (teacher model)
│   ├── prepare_egoblind_data.py       # dataset preprocessing + urgency tagging
│   ├── inference.py                   # single-model inference with tag routing
│   ├── generate_dpo_pairs.py          # DPO pair construction with composite loss
│   ├── kimi_api_baseline.py           # Kimi API baseline evaluation
│   ├── aws_setup.sh                   # AWS spot instance setup
│   └── remote_setup.sh                # remote environment setup
├── results/
│   └── baseline/
│       ├── all_results.json
│       ├── baseline_predictions.json
│       ├── best_config_breakdown.json
│       ├── top_20_results.json
│       └── summary_table.tsv
├── LICENSE
└── README.md
```

## Data
The dataset is derived from the **EgoBlind** project (Xiao et al., 2025).  As described [here](https://github.com/doc-doc/EgoBlind), the data is characterised by: 

- **Regions of Interest (ROI):** Often off-center and not well-focused, reflecting the challenges of egocentric visual perception.  
- **Questions:** Formulated to capture user intention in specific situations. They may be ambiguous if spatial and temporal context is ignored.  
- **Answers:** Must be not only visually correct but also contextually helpful to the user, considering both space and time.  

![egoblind](https://github.com/user-attachments/assets/eaeae917-ffab-47f3-a68d-cca74118fcde)
*(Source: [EgoBlind GitHub](https://github.com/doc-doc/EgoBlind))*

## Pipeline

### 1. Urgency annotation (teacher)

`classify_urgency.py` uses GPT to label every question–video pair in EgoBlind as `urgent` or `non-urgent` based on immediate physical safety implications for a blind user. A statistically significant subset is manually validated for agreement.

### 2. Urgency classifier (student)

`models/CLIP_Urgency_Classifier.ipynb` trains a lightweight CLIP-based binary classifier on the annotated labels. The model encodes both the video (sampled frames via ViT) and the text query, fuses the representations, and predicts urgency. Evaluated with F1 to balance precision and recall — false negatives in safety-critical cases are especially costly.

### 3. Routing + response policies

At inference time, the urgency classifier routes each query:

```
EgoBlind query
      │
      ▼
CLIP urgency classifier
      │
   urgent?
   /      \
  yes      no
  │         │
low-latency  best-of-k
 policy      policy
      │         │
      └────┬────┘
           ▼
     EgoBlind metrics
```

**Approach 1 (Xander):** hard-routes to two separate fine-tuned LoRA adapters with regime-specific loss functions.

**Approach 2 (Julia):** appends `[URGENT]` or `[NON-URGENT]` tag to a single LoRA adapter trained on a mixed objective.

## Metrics

| Component | Metric |
|---|---|
| Urgency classifier | F1 |
| Generative model | Accuracy + utility score (EgoBlind) |
| Urgent responses | Latency penalty (generation time + verbosity) |

## Reproducing

### Prerequisites

```bash
pip install openai torch torchvision transformers
export MOONSHOT_API_KEY="your-key-here"
```

### Phase 1: Baseline (laptop)

```bash
python scripts/kimi_api_baseline.py \
    --test_data /path/to/egoblind_test.json \
    --output_path results/baseline/baseline_predictions.json
```

### Phase 2: Training (AWS)

```bash
# 1. Launch and set up instance
bash scripts/aws_setup.sh
ssh -i ~/.ssh/your-key.pem ubuntu@<IP>
bash scripts/remote_setup.sh

# 2. Upload data and configs
scp -r data/ configs/ scripts/ ubuntu@<IP>:~/LLaMA-Factory/

# 3. Prepare data with urgency labels and tags
python scripts/prepare_egoblind_data.py \
    --egoblind_dir /path/to/egoblind \
    --urgency_labels /path/to/urgency_labels.json \
    --output_dir data/

# 4. SFT
llamafactory-cli train configs/sft_unified.yaml   # Approach 2 (Julia)

# 5. Generate DPO pairs
python scripts/generate_dpo_pairs.py \
    --adapter_path output/sft_unified/checkpoint-final \
    --data_path data/egoblind_urgent_for_dpo.json \
    --output_path data/egoblind_unified_dpo.json \
    --alpha 0.4 --beta 0.3 --gamma 0.3

# 6. DPO training
llamafactory-cli train configs/dpo_unified.yaml

# 7. Download checkpoint before terminating
scp -r ubuntu@<IP>:~/LLaMA-Factory/output/ ./output/
```

> ⚠️ **Important:** terminate your AWS instance when done. Download the adapter checkpoint (~200–400MB) before shutting down.

### Phase 3: Evaluation

```bash
python scripts/inference.py \
    --adapter_path output/dpo_unified/checkpoint-final \
    --test_data /path/to/egoblind_test.json \
    --output_path output/unified_predictions.json
```

## Results

Baseline results are in `results/baseline/`. See `summary_table.tsv` for a full breakdown and `best_config_breakdown.json` for the top configuration analysis.

## References

- [EgoBlind: Towards Egocentric Visual Assistance for the Blind](https://arxiv.org/abs/2503.08221) (Xiao et al., 2025)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (Radford et al., 2021)
- [VizWiz](https://dl.acm.org/doi/10.1145/1866029.1866080) (Bigham et al., 2010)
- [Ego4D](https://arxiv.org/abs/2110.07058) (Grauman et al., 2021)
