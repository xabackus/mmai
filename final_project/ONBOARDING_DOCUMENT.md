# EgoBlind-RA: Complete Project Onboarding Document
## For Session Continuity — April 2026

---

## 1. PROJECT OVERVIEW

**Project name:** EgoBlind-RA (Risk-Adaptive Routing)  
**Course:** Modeling: Multimodal AI (MAS.S60 / 6.S985) at MIT  
**Team:** Xander Backus (bifurcated model approach) and Julia Kim (prompt-conditioned approach)  
**GitHub:** github.com/juliavekim/EgoBlind-RA  

**Core idea:** Classify blind users' egocentric video questions as urgent vs. not-urgent, then route them to different response policies. Urgent queries get fast, concise responses; non-urgent get detailed, thorough responses. Compare two architectures:
- **Approach 1 (Xander):** Two separate fine-tuned LoRA adapters (one for urgent, one for non-urgent)
- **Approach 2 (Julia):** One model with [URGENT]/[NON-URGENT] tags prepended to the system prompt

**Inspirational paper:** "EgoBlind: Towards Egocentric Visual Assistance for the Blind People" (Xiao et al., 2025, arXiv:2503.08221)

---

## 2. THE DATASET

**Source:** EgoBlind dataset — egocentric videos from real blind users + questions they'd ask.

### File structure (on Xander's MacBook, in ~/MMAI_Project/):
- `train_labeled.csv` — 2,746 rows
  - Columns: `video_name, question_id, question, answer0, answer1, answer2, answer3, type, start-time/s, urgency`
  - `video_name` is 5-digit zero-padded (e.g., `00000`)
  - `urgency` column has values: `urgent`, `not_urgent`, or `ERROR` (a few bad rows)
  - 1,474 urgent, 1,272 not_urgent in train
- `test_labeled.csv` — 2,565 total rows, but only 1,283 have answers
  - Combined from two files with different column schemas (`test_half_hold.csv` has no answers, `test_half_release.csv` has answers)
  - `video_name` in test CSVs is NOT zero-padded, but video files ARE (use `.zfill(5)`)
  - 668 urgent, 615 not_urgent in test (among rows with answers)
- `train_videos/` — 922 videos (00000.mp4 through 00921.mp4), 35GB total
- `test_videos/` — test videos, 32GB total

### Urgency labels
Generated earlier in the semester using GPT-5.2 (model: `gpt-5.2-2025-12-11`) with `reasoning_effort="xhigh"` and `max_completion_tokens=2048`. 5 frames extracted per video, sent as base64 JPEG. Labels are in the `urgency` column of both CSVs already.

### Question type categories (from EgoBlind paper):
- Information Reading (largest category)
- Safety Warnings
- Navigation
- Social Communication
- Tool Use
- Other Resources

---

## 3. THE LOSS FUNCTION

### Urgent queries:
```
L_urgent = α·L_acc + β·L_util + γ·L_lat
```

### Non-urgent queries:
```
L_nonurgent = α·L_acc + β·L_util
```

### Components:
- **L_acc** = 1 − max_k S(y, y*_k) — semantic similarity (token-overlap F1) against best reference answer
- **L_util** = 1 − S(y, y*) — same as accuracy for now (proxy; ideally a separate LLM judge)
- **L_lat** = piecewise on output token count:
  - 0 if |y| ≤ τ_min
  - ((|y| − τ_min) / (τ_max − τ_min))^p if τ_min < |y| ≤ τ_max
  - 1 + κ·(|y| − τ_max) if |y| > τ_max

### Default hyperparameters:
α=0.4, β=0.3, γ=0.3, τ_min=5, τ_max=30, p=2, κ=0.5

### Grid search best (by test loss):
α=0.2, β=0.1, γ=0.1, τ_min=8, τ_max=40, p=3, κ=0.3

### LaTeX writeup exists:
`loss_function.pdf` and `loss_function.tex` — 2-page formal writeup of the loss function with all equations and hyperparameter table.

---

## 4. WHAT HAS BEEN COMPLETED

### 4a. Urgency classification (earlier in semester)
- All train (2,746) and test (2,565) questions classified as urgent/not_urgent
- Done with GPT-5.2 via OpenAI API with video frames
- Labels saved as `urgency` column in CSVs

### 4b. Baseline evaluation (completed)
- **Model used:** `moonshot-v1-32k-vision-preview` via Moonshot API
- **Input:** 5 frames per video (320px wide, JPEG quality 60, extracted on-the-fly from local videos) + question text
- **Prompts:** Different system prompts for urgent vs. non-urgent
- **Cost:** $32.10 of $50 Kimi credits, 21.4M tokens, ZERO errors
- **All 4,029 examples** processed (2,746 train + 1,283 test)
- **Results saved to:** `baseline_predictions.json` (IRREPLACEABLE — no credits left to re-run)

### Baseline results:
| Metric | Overall | Urgent | Not_urgent |
|--------|---------|--------|------------|
| Token-overlap F1 | 0.1025 | 0.1245 | 0.0778 |
| Test examples | 4,029 | 2,142 | 1,887 |

### Per-type baseline (from grid search, test set, best config):
| Type | Loss | Similarity | N |
|------|------|-----------|---|
| safety warnings | 1.1184 | 0.1182 | 946 |
| tool use | 0.9476 | 0.1386 | 211 |
| navigation | 0.7865 | 0.1172 | 363 |
| information reading | 0.4891 | 0.0885 | 1873 |
| other resources | 0.5010 | 0.1122 | 222 |
| social communication | 0.4589 | 0.0648 | 37 |
| communication and interaction | 0.4504 | 0.0653 | 81 |

**Key finding:** Safety warnings and tool use have the highest loss — models struggle most on exactly the queries where getting it wrong is most dangerous. This validates the entire project premise.

### 4c. Grid search (completed)
- Scored all 4,029 baseline predictions across 3,888 hyperparameter combinations
- Results saved in `results/` directory:
  - `all_results.json` — full grid search
  - `top_20_results.json` — best 20 configs
  - `summary_table.tsv` — top 50 as spreadsheet
  - `best_config_breakdown.json` — per-type detail for best config

### 4d. SFT fine-tuning (completed, BUT MODELS ARE BROKEN)
- **Infrastructure:** MIT Engaging cluster (SLURM), L40S GPUs (48GB)
- **Base model:** moonshotai/Kimi-VL-A3B-Instruct (16B total, 2.8B active, MoE)
- **Method:** LoRA via LLaMA-Factory
- **LoRA config:** rank=16, alpha=32, targets=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
- **Training:** 5 epochs, batch_size=1, grad_accum=16, lr=2e-5, cosine schedule
- **TEXT ONLY** — no vision/video frames during training (only text question → text answer)

#### Urgent SFT:
- 1,474 training examples (shortest correct reference answer as target)
- 465 steps, 3h 46m
- Loss: 1.83 → 0.48
- Saved at: `~/LLaMA-Factory/output/sft_urgent/` on Engaging
- Backed up at: `~/sft_urgent_backup/` on Engaging

#### Non-urgent SFT:
- 1,272 training examples (longest correct reference answer as target)
- 400 steps, 3h 05m
- Loss: 1.61 → 1.22
- Saved at: `~/LLaMA-Factory/output/sft_nonurgent/` on Engaging
- Backed up at: `~/sft_nonurgent_backup/` on Engaging

### 4e. Evaluation (completed, BUT RESULTS ARE BAD)
- Ran both fine-tuned models on 1,283 test examples
- Results saved to: `~/eval_results.json` and `~/eval_summary.json` on Engaging

#### Evaluation results (BROKEN):
| Metric | Baseline | Ours | Delta |
|--------|----------|------|-------|
| Overall loss | 3.2470 | 4.5697 | +1.3227 (worse) |
| Overall similarity | 0.1028 | 0.0944 | -0.0084 (worse) |
| Urgent loss | 5.6436 | 8.2001 | +2.5565 (much worse) |
| Not_urgent loss | 0.6437 | 0.6262 | -0.0175 (slightly better) |

**CRITICAL: Both models generate garbage text** — repeated tokens, Chinese characters, degenerate loops like "I don't know. I don't know. I don't know." or ":No.:Yes.:Yes.:Yes.:Yes."

---

## 5. WHAT WENT WRONG WITH FINE-TUNING

### Root cause (most likely): Missing `template` in LLaMA-Factory config

The `sft_urgent.yaml` and `sft_nonurgent.yaml` configs did NOT specify a `template` parameter. LLaMA-Factory uses templates to format training data into the correct chat format for each model (special tokens, role markers, etc.). Without it, LLaMA-Factory likely used a default template that doesn't match Kimi-VL's actual tokenization.

This means the model was trained on incorrectly formatted text. At inference time, when we used Kimi-VL's native `processor.apply_chat_template()`, the model saw a format it was never trained on and produced garbage.

### Fix needed:
1. Check if LLaMA-Factory has a registered template for Kimi-VL: `grep -r "kimi" ~/LLaMA-Factory/src/llamafactory/data/template.py`
2. If not, register a custom template matching Kimi-VL's chat format
3. Re-run SFT with the correct template
4. Re-evaluate

### Other issues encountered during setup:
- **transformers version conflicts:** LLaMA-Factory requires >=4.55.0, Kimi-VL model code was written for 4.48.2
- **DynamicCache API changes:** `seen_tokens` → `get_seq_length()`, `get_usable_length()` → `get_seq_length()`
- **Gradient checkpointing:** KimiVLForConditionalGeneration doesn't support it — had to disable
- **OOM:** rank-64 LoRA was too large (1.1B trainable params) — reduced to rank-16 (still large but fits)
- **Two conda environments needed:** `egoblind` (transformers 4.55.0 for LLaMA-Factory training) and `egoblind-eval` (transformers 4.48.2 for Kimi-VL inference)

---

## 6. INFRASTRUCTURE DETAILS

### MIT Engaging cluster:
- **SSH access:** Must hop through Athena — `ssh xabackus@athena.dialup.mit.edu` then `ssh xabackus@orcd-login.mit.edu` (direct SSH doesn't work from off-campus or MIT SECURE wifi)
- **GPU partition:** `mit_normal_gpu` — has H200 (8x), H100 (4x), L40S (4x per node, 50 nodes)
- **Time limit:** 6 hours per job
- **Storage:** 187TB available on /home, model cached at `~/.cache/huggingface/` (~31GB)
- **Conda:** `module load miniforge` then `conda activate egoblind` (training) or `conda activate egoblind-eval` (inference)
- **SLURM:** Submit with `sbatch`, monitor with `squeue --me`, check output in `~/LLaMA-Factory/logs/`

### Two conda environments on Engaging:
| Environment | Python | Transformers | Purpose |
|-------------|--------|-------------|---------|
| egoblind | 3.11 | 4.55.0 | LLaMA-Factory SFT/DPO training |
| egoblind-eval | 3.10 | 4.48.2 | Kimi-VL inference/evaluation |

### Moonshot API:
- **Base URL:** https://api.moonshot.ai/v1 (OpenAI-compatible)
- **Client:** `from openai import OpenAI; client = OpenAI(api_key=KEY, base_url="https://api.moonshot.ai/v1")`
- **Vision model used for baseline:** `moonshot-v1-32k-vision-preview`
- **Available models:** moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k, *-vision-preview variants, kimi-k2-*, kimi-k2.5
- **Credits remaining:** ~$18 (spent $32.10 on baseline)
- **API key was rotated** — old one was accidentally posted publicly

---

## 7. FILES ON XANDER'S MACBOOK (~/MMAI_Project/)

### Critical (irreplaceable):
- `baseline_predictions.json` — 4,029 API predictions ($32 worth)
- `baseline_checkpoint.json` — same data, checkpoint format
- `train_labeled.csv` — training data with urgency labels
- `test_labeled.csv` — test data with urgency labels
- `train_videos/` — 35GB of training videos
- `test_videos/` — 32GB of test videos

### Results:
- `results/all_results.json` — 3,888 grid search results
- `results/top_20_results.json` — best 20 hyperparameter configs
- `results/summary_table.tsv` — top 50 as spreadsheet
- `results/best_config_breakdown.json` — best config detail
- `engaging_results/` — evaluation results, training logs, loss curves from Engaging

### Scripts:
- `run_baseline.py` — Kimi API baseline (with vision, on-the-fly frame extraction)
- `step2_grid_search.py` — offline scoring across hyperparameter grid
- `save_frames.py` — extract and save frames for Julia
- `extract_frames.py` — earlier frame extraction attempt (killed, incomplete)

### Documents:
- `loss_function.pdf` / `loss_function.tex` — formal loss function writeup

---

## 8. FILES ON ENGAGING CLUSTER (xabackus@orcd-login.mit.edu)

### Home directory (~):
- `train_labeled.csv`, `test_labeled.csv`, `baseline_predictions.json` — copies of data
- `eval_results.json` — 1,283 test predictions from fine-tuned models (broken outputs)
- `eval_summary.json` — summary comparison table
- `evaluate.py` — evaluation script
- `run_eval.sh` — SLURM job script for evaluation
- `run_sft_urgent.sh`, `run_sft_nonurgent.sh` — SLURM job scripts for training
- `prepare_data.py` — data prep script
- `sft_urgent_backup/` — backup of urgent adapter
- `sft_nonurgent_backup/` — backup of non-urgent adapter

### ~/LLaMA-Factory/:
- `configs/sft_urgent.yaml` — urgent training config (MISSING template parameter)
- `configs/sft_nonurgent.yaml` — non-urgent training config (MISSING template parameter)
- `data/egoblind_urgent_sft.json` — 1,474 urgent training examples (sharegpt format)
- `data/egoblind_nonurgent_sft.json` — 1,272 non-urgent training examples (sharegpt format)
- `output/sft_urgent/` — urgent adapter + checkpoints
- `output/sft_nonurgent/` — non-urgent adapter + checkpoints
- `logs/` — all SLURM job outputs

---

## 9. TRAINING PIPELINE (for reference / re-running)

### Step 1: Data prep
```bash
python ~/prepare_data.py
```
Creates `egoblind_urgent_sft.json` (1,474 examples, shortest answers) and `egoblind_nonurgent_sft.json` (1,272 examples, longest answers) in LLaMA-Factory sharegpt format.

### Step 2: SFT
```bash
module load miniforge && conda activate egoblind
cd ~/LLaMA-Factory
llamafactory-cli train configs/sft_urgent.yaml
```

### Step 3: Evaluate
```bash
module load miniforge && conda activate egoblind-eval
cd ~/LLaMA-Factory
sbatch ~/run_eval.sh
```

---

## 10. DPO PIPELINE (NOT YET IMPLEMENTED)

The plan was:
1. Generate 8 candidate responses per urgent example from the SFT model
2. Score each with composite loss (α·L_acc + β·L_util + γ·L_lat)
3. Pick lowest-loss as "chosen", highest-loss as "rejected"
4. Train with DPO (LLaMA-Factory supports `stage: dpo`)

Scripts were written (`generate_dpo_pairs.py`, `dpo_urgent.yaml`) but never run because the SFT models produce garbage.

---

## 11. NEXT STEPS (PRIORITY ORDER)

### For the midterm report (due April 7):
1. **Write the paper** — we have enough for a midterm even with broken fine-tuning:
   - Baseline results are solid and interesting
   - Grid search shows hyperparameter sensitivity
   - Training loss curves show the model learned *something* (just with wrong template)
   - The failure itself is an interesting discussion point
   - Frame this as: "initial experiments revealed a template mismatch issue"

### After midterm:
2. **Fix the template issue** — investigate LLaMA-Factory's template registry for Kimi-VL
3. **Re-run SFT** with correct template (~4 hrs per model on Engaging)
4. **Re-evaluate** — should produce coherent outputs
5. **Run DPO** on the urgent model if SFT results are good
6. **Julia's approach** — she needs to run her prompt-conditioned variant for comparison
7. **Final evaluation** with proper semantic similarity (BERTScore or LLM judge, not just token overlap)

---

## 12. MIDTERM REPORT STRUCTURE (NeurIPS format, 6 pages + refs)

### What we have content for:
- **Abstract:** ✅ Can write from project description
- **Introduction:** ✅ EgoBlind paper motivation, 2.2B visually impaired people, models struggle on safety queries
- **Related work:** ✅ EgoBlind, VizWiz, Ego4D, adaptive computation, DPO literature
- **Problem statement:** ✅ Loss function is fully formalized in LaTeX
- **Proposed approach:** ✅ Two architectures described in detail, loss function, DPO pipeline
- **Experimental methodology:** ✅ Dataset details, urgency annotation, baseline setup
- **Results and discussion:** ⚠️ Baseline results are good; fine-tuning results are broken but can be discussed as lessons learned
- **Next steps:** ✅ Fix template, re-run SFT, DPO, Julia's approach, better eval metrics

### Figures/tables we can make:
1. System architecture diagram (classifier → router → urgent/non-urgent models)
2. Baseline performance table by question type (safety warnings worst, confirming EgoBlind)
3. Grid search heatmap or table showing hyperparameter sensitivity
4. Training loss curves for both SFT models (loss dropping is good, even if outputs are broken)
5. Comparison table: baseline vs. fine-tuned (showing the failure, with discussion)
6. Example predictions showing garbage outputs (failure analysis)
7. Loss function formalization (from the LaTeX doc)

---

## 13. KEY NUMBERS TO REMEMBER

| Metric | Value |
|--------|-------|
| Total train examples | 2,746 |
| Total test examples (with answers) | 1,283 |
| Urgent train | 1,474 |
| Not_urgent train | 1,272 |
| Urgent test | 668 |
| Not_urgent test | 615 |
| Baseline API cost | $32.10 |
| Baseline tokens | 21.4M |
| Baseline overall F1 | 0.1025 |
| Urgent SFT training time | 3h 46m |
| Non-urgent SFT training time | 3h 05m |
| Urgent SFT final loss | 0.48 |
| Non-urgent SFT final loss | 1.22 |
| LoRA rank used | 16 |
| LoRA trainable params | ~287M (after reducing from 1.1B) |
| Kimi-VL total params | 17.5B |
| Kimi-VL active params | 2.8B |

---

## 14. BUGS AND GOTCHAS LOG

| Issue | Solution |
|-------|----------|
| `max_tokens` not supported by GPT-5.2 | Use `max_completion_tokens` |
| `temperature=0` not supported by GPT-5.2 | Remove temperature parameter |
| Everything classified as not_urgent | Bumped max_completion_tokens to 2048, improved prompt |
| Test video names not matching | `.zfill(5)` to zero-pad |
| Test CSV column mismatch | Two CSVs have different schemas, merged with `dict.fromkeys()` |
| Grid search KeyError 'non-urgent' | Data uses `not_urgent` (underscore), not `non-urgent` (hyphen) |
| Grid search KeyError 'ERROR' | Some urgency labels are 'ERROR', made dict dynamic |
| SSH to Engaging hangs | Must hop through Athena: `ssh athena` then `ssh orcd-login` |
| `conda: command not found` on Engaging | Run `module load miniforge` first |
| LLaMA-Factory requires Python >=3.11 | Recreated conda env with Python 3.11 |
| LLaMA-Factory requires transformers >=4.55.0 | Installed 4.55.0 (but conflicts with Kimi-VL's 4.48.2) |
| KimiVL doesn't support gradient checkpointing | Set `gradient_checkpointing: false` |
| OOM with rank-64 LoRA | Reduced to rank-16, batch_size=1 |
| `resume_from_checkpoint: true` fails on first run | Set to false for first run |
| DynamicCache `seen_tokens` error | `sed` replace with `get_seq_length()` in cached model code |
| DynamicCache `get_usable_length` error | `sed` replace with `get_seq_length()` in cached model code |
| Attention mask size mismatch | Deeper transformers/Kimi-VL incompatibility — solved by using separate conda envs |
| tiktoken not installed in eval env | `pip install tiktoken` |
| Eval job hit 2hr time limit | Increased to 5:50 in SLURM script |
| SFT models produce garbage at inference | **UNSOLVED** — likely missing `template` parameter in LLaMA-Factory config |
| Athena AFS quota 2GB | Too small for videos; must use frames or run from laptop |
| API key accidentally posted | Rotated immediately on platform.moonshot.ai |
