"""
rescore_pipeline_canonical.py

Re-score the pipeline evaluation results with canonical paper params:
  α=0.5, β=0.3, γ=0.2, τ_min=8, τ_max=40, p=3, κ=0.3

No GPU needed — just loads JSON and recomputes.
"""

import json, csv, os

# Canonical params (paper)
ALPHA = 0.5
BETA = 0.3
GAMMA = 0.2
TAU_MIN = 8
TAU_MAX = 40
P_EXP = 3
KAPPA = 0.3

PIPELINE_PATH = os.path.expanduser("~/FINAL_eval_pipeline_clip.json")
THREE_COND_PATH = os.path.expanduser("~/FINAL_eval_three_conditions.json")
TEST_CSV = os.path.expanduser("~/test_labeled.csv")


def semantic_similarity(pred, refs):
    cand = set(pred.lower().split())
    if not cand:
        return 0.0
    best = 0.0
    for ref in refs:
        r = set(ref.lower().split())
        if not r:
            continue
        o = cand & r
        p = len(o) / len(cand)
        rc = len(o) / len(r)
        if p + rc > 0:
            best = max(best, 2 * p * rc / (p + rc))
    return best


def composite_loss(pred, refs, is_urgent):
    sim = semantic_similarity(pred, refs)
    la = 1.0 - sim
    lu = 1.0 - sim
    length = len(pred.split())
    if is_urgent:
        if length <= TAU_MIN:
            ll = 0.0
        elif length <= TAU_MAX:
            ll = ((length - TAU_MIN) / (TAU_MAX - TAU_MIN)) ** P_EXP
        else:
            ll = 1.0 + KAPPA * (length - TAU_MAX)
        return ALPHA * la + BETA * lu + GAMMA * ll, sim, length
    else:
        # Renormalize: α/(α+β) and β/(α+β)
        a_norm = ALPHA / (ALPHA + BETA)
        b_norm = BETA / (ALPHA + BETA)
        return a_norm * la + b_norm * lu, sim, length


def load_test_data():
    rows = {}
    with open(TEST_CSV) as f:
        for row in csv.DictReader(f):
            qid = row.get("question_id", "").strip()
            answers = [row.get(f"answer{i}", "").strip()
                       for i in range(4)
                       if row.get(f"answer{i}", "").strip()]
            rows[qid] = {
                "answers": answers,
                "urgency": row.get("urgency", "not_urgent").strip(),
            }
    return rows


def score_predictions(predictions, test_data):
    """Score a dict of {qid: prediction_text} against test data."""
    losses, sims, lengths = [], [], []
    for qid, pred in predictions.items():
        if qid not in test_data:
            continue
        td = test_data[qid]
        is_urgent = td["urgency"] == "urgent"
        loss, sim, length = composite_loss(pred, td["answers"], is_urgent)
        losses.append(loss)
        sims.append(sim)
        lengths.append(length)
    return {
        "loss": round(sum(losses) / len(losses), 4),
        "similarity": round(sum(sims) / len(sims), 4),
        "mean_words": round(sum(lengths) / len(lengths), 1),
        "n": len(losses),
    }


def main():
    print("=" * 60)
    print("RE-SCORING WITH CANONICAL PARAMS")
    print(f"α={ALPHA}, β={BETA}, γ={GAMMA}")
    print(f"τ_min={TAU_MIN}, τ_max={TAU_MAX}, p={P_EXP}, κ={KAPPA}")
    print("=" * 60)

    test_data = load_test_data()
    print(f"\nLoaded {len(test_data)} test examples")

    # --- CLIP pipeline results ---
    print("\n--- CLIP Pipeline (FINAL_eval_pipeline_clip.json) ---")
    with open(PIPELINE_PATH) as f:
        pipeline = json.load(f)

    details = pipeline["details"]

    # CLIP pipeline: use the routed prediction
    clip_preds = {}
    for item in details:
        clip_preds[item["question_id"]] = item["prediction"]

    clip_metrics = score_predictions(clip_preds, test_data)
    print(f"CLIP pipeline:  {clip_metrics}")

    # Oracle: use ground-truth urgency to select prediction
    oracle_preds = {}
    for item in details:
        qid = item["question_id"]
        td = test_data.get(qid)
        if not td:
            continue
        if td["urgency"] == "urgent":
            oracle_preds[qid] = item["urgent_prediction"]
        else:
            oracle_preds[qid] = item["nonurgent_prediction"]

    oracle_metrics = score_predictions(oracle_preds, test_data)
    print(f"Oracle routing: {oracle_metrics}")

    # --- Three-conditions results (for zero-shot pipeline) ---
    print("\n--- Zero-Shot Pipeline (FINAL_eval_three_conditions.json) ---")
    with open(THREE_COND_PATH) as f:
        three_cond = json.load(f)

    # Zero-shot pipeline = condition3
    cond3 = three_cond["details"]["condition3"]
    zs_preds = {}
    for item in cond3:
        zs_preds[item["question_id"]] = item["prediction"]

    zs_metrics = score_predictions(zs_preds, test_data)
    print(f"Zero-shot:      {zs_metrics}")

    # --- Print table for paper ---
    print(f"\n{'='*60}")
    print("TABLE FOR PAPER (canonical params)")
    print(f"{'='*60}")
    print(f"{'Routing':<30} {'Loss':>8} {'Similarity':>12} {'Mean words':>12}")
    print(f"{'-'*62}")
    print(f"{'Baseline (uniform policy)':<30} {'0.607':>8} {'0.350':>12} {'84.1':>12}")
    print(f"{'Zero-shot (prompted Kimi-VL)':<30} {zs_metrics['loss']:>8.4f} {zs_metrics['similarity']:>12.4f} {zs_metrics['mean_words']:>12.1f}")
    print(f"{'CLIP classifier':<30} {clip_metrics['loss']:>8.4f} {clip_metrics['similarity']:>12.4f} {clip_metrics['mean_words']:>12.1f}")
    print(f"{'Oracle (ground-truth labels)':<30} {oracle_metrics['loss']:>8.4f} {oracle_metrics['similarity']:>12.4f} {oracle_metrics['mean_words']:>12.1f}")

    gap_old = oracle_metrics['similarity'] - zs_metrics['similarity']
    gap_new = oracle_metrics['similarity'] - clip_metrics['similarity']
    if gap_old > 0:
        pct = (1 - gap_new / gap_old) * 100
        print(f"\nGap closed: {pct:.1f}%")


if __name__ == "__main__":
    main()
