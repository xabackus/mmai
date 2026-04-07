"""
step2_grid_search.py

Scores all saved predictions across a grid of hyperparameter combinations.
NO API calls — runs purely on the saved predictions from step 1.
This is free and fast. Re-run as many times as you want.

Usage:
    python step2_grid_search.py \
        --predictions results/raw_predictions.json \
        --output_dir results/grid_search/
"""

import argparse
import json
import os
import itertools
from pathlib import Path


# ============================================================
# Loss Components
# ============================================================

def semantic_similarity(candidate: str, references: list[str]) -> float:
    """Max token-overlap F1 against any reference."""
    cand_tokens = set(candidate.lower().split())
    if not cand_tokens:
        return 0.0
    best = 0.0
    for ref in references:
        ref_tokens = set(ref.lower().split())
        if not ref_tokens:
            continue
        overlap = cand_tokens & ref_tokens
        p = len(overlap) / len(cand_tokens)
        r = len(overlap) / len(ref_tokens)
        if p + r > 0:
            best = max(best, 2 * p * r / (p + r))
    return best


def loss_accuracy(prediction: str, references: list[str]) -> float:
    return 1.0 - semantic_similarity(prediction, references)


def loss_utility(prediction: str, references: list[str]) -> float:
    """Proxy: same as accuracy for baseline. Replace with LLM judge for final eval."""
    return 1.0 - semantic_similarity(prediction, references)


def loss_latency(
    prediction: str,
    tau_min: int, tau_max: int, p: float, kappa: float,
) -> float:
    length = len(prediction.split())
    if length <= tau_min:
        return 0.0
    elif length <= tau_max:
        return ((length - tau_min) / (tau_max - tau_min)) ** p
    else:
        return 1.0 + kappa * (length - tau_max)


def compute_loss(
    prediction: str, references: list[str], is_urgent: bool,
    alpha: float, beta: float, gamma: float,
    tau_min: int, tau_max: int, p: float, kappa: float,
) -> dict:
    """
    Compute loss for a single example.
    Urgent: alpha*L_acc + beta*L_util + gamma*L_lat
    Non-urgent: alpha*L_acc + beta*L_util
    """
    l_acc = loss_accuracy(prediction, references)
    l_util = loss_utility(prediction, references)
    l_lat = loss_latency(prediction, tau_min, tau_max, p, kappa) if is_urgent else 0.0

    if is_urgent:
        total = alpha * l_acc + beta * l_util + gamma * l_lat
    else:
        total = alpha * l_acc + beta * l_util

    return {
        "l_acc": round(l_acc, 4),
        "l_util": round(l_util, 4),
        "l_lat": round(l_lat, 4),
        "l_total": round(total, 4),
        "pred_length": len(prediction.split()),
        "sim": round(1.0 - l_acc, 4),
    }


# ============================================================
# Grid Search
# ============================================================

# Hyperparameter grid
GRID = {
    "alpha": [0.2, 0.3, 0.4, 0.5],
    "beta":  [0.1, 0.2, 0.3],
    "gamma": [0.1, 0.2, 0.3, 0.4],
    "tau_min": [3, 5, 8],
    "tau_max": [20, 30, 40],
    "p": [1, 2, 3],
    "kappa": [0.3, 0.5, 1.0],
}


def generate_grid_combos(grid: dict) -> list[dict]:
    """Generate all hyperparameter combinations."""
    keys = sorted(grid.keys())
    combos = []
    for values in itertools.product(*[grid[k] for k in keys]):
        combo = dict(zip(keys, values))
        # Only keep combos where alpha + beta + gamma is reasonable
        # (they don't need to sum to 1, but filter degenerate cases)
        if combo["alpha"] + combo["beta"] + combo["gamma"] > 0:
            combos.append(combo)
    return combos


def score_all_predictions(predictions: list[dict], params: dict) -> dict:
    """Score all predictions with a single hyperparameter combo. Returns summary stats."""
    results_by_split = {"train": [], "test": []}
    results_by_type = {}
    results_by_urgency = {}

    for pred in predictions:
        if pred["status"] != "ok":
            continue

        is_urgent = pred["urgency"] == "urgent"
        loss_result = compute_loss(
            pred["prediction"], pred["answers"], is_urgent,
            alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"],
            tau_min=params["tau_min"], tau_max=params["tau_max"],
            p=params["p"], kappa=params["kappa"],
        )

        entry = {**loss_result, "urgency": pred["urgency"], "question_type": pred["question_type"]}

        results_by_split[pred["split"]].append(entry)
        urg = pred["urgency"]
        if urg not in results_by_urgency:
            results_by_urgency[urg] = []
        results_by_urgency[urg].append(entry)

        q_type = pred["question_type"] or "unknown"
        if q_type not in results_by_type:
            results_by_type[q_type] = []
        results_by_type[q_type].append(entry)

    def summarize(entries: list[dict]) -> dict:
        if not entries:
            return {"n": 0}
        n = len(entries)
        return {
            "n": n,
            "avg_loss": round(sum(e["l_total"] for e in entries) / n, 4),
            "avg_acc_loss": round(sum(e["l_acc"] for e in entries) / n, 4),
            "avg_util_loss": round(sum(e["l_util"] for e in entries) / n, 4),
            "avg_lat_loss": round(sum(e["l_lat"] for e in entries) / n, 4),
            "avg_similarity": round(sum(e["sim"] for e in entries) / n, 4),
            "avg_pred_length": round(sum(e["pred_length"] for e in entries) / n, 1),
        }

    return {
        "params": params,
        "overall": summarize(results_by_split["train"] + results_by_split["test"]),
        "by_split": {k: summarize(v) for k, v in results_by_split.items()},
        "by_urgency": {k: summarize(v) for k, v in results_by_urgency.items()},
        "by_type": {k: summarize(v) for k, v in sorted(results_by_type.items())},
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to raw_predictions.json from step 1")
    parser.add_argument("--output_dir", default="results/grid_search/")
    parser.add_argument("--custom_grid", default=None, help="Optional JSON file with custom grid values")
    args = parser.parse_args()

    # Load predictions
    with open(args.predictions, "r") as f:
        predictions = json.load(f)

    ok_count = sum(1 for p in predictions if p["status"] == "ok")
    print(f"Loaded {len(predictions)} predictions ({ok_count} successful)")

    # Load custom grid if provided
    grid = GRID
    if args.custom_grid:
        with open(args.custom_grid, "r") as f:
            grid = json.load(f)
        print(f"Using custom grid from {args.custom_grid}")

    # Generate combos
    combos = generate_grid_combos(grid)
    print(f"Grid search: {len(combos)} hyperparameter combinations")

    os.makedirs(args.output_dir, exist_ok=True)

    # Run grid search
    all_results = []
    best_result = None
    best_loss = float("inf")

    for i, params in enumerate(combos):
        result = score_all_predictions(predictions, params)
        all_results.append(result)

        # Track best (lowest overall loss on test set)
        test_loss = result["by_split"]["test"].get("avg_loss", float("inf"))
        if test_loss < best_loss:
            best_loss = test_loss
            best_result = result

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(combos)}] current best test loss: {best_loss:.4f}")

    # Sort by test loss
    all_results.sort(key=lambda r: r["by_split"]["test"].get("avg_loss", float("inf")))

    # Save full results
    full_path = os.path.join(args.output_dir, "all_results.json")
    with open(full_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save top 20
    top_path = os.path.join(args.output_dir, "top_20_results.json")
    with open(top_path, "w") as f:
        json.dump(all_results[:20], f, indent=2)

    # Save summary table (TSV for easy viewing)
    table_path = os.path.join(args.output_dir, "summary_table.tsv")
    with open(table_path, "w") as f:
        headers = [
            "rank", "alpha", "beta", "gamma", "tau_min", "tau_max", "p", "kappa",
            "test_loss", "train_loss",
            "test_sim", "train_sim",
            "test_avg_len", "train_avg_len",
            "urgent_loss", "nonurgent_loss",
        ]
        f.write("\t".join(headers) + "\n")

        for rank, r in enumerate(all_results[:50], 1):
            p = r["params"]
            ts = r["by_split"]["test"]
            tr = r["by_split"]["train"]
            urg = r["by_urgency"]["urgent"]
            nonurg = r["by_urgency"]["not_urgent"]
            row = [
                rank,
                p["alpha"], p["beta"], p["gamma"],
                p["tau_min"], p["tau_max"], p["p"], p["kappa"],
                ts.get("avg_loss", ""), tr.get("avg_loss", ""),
                ts.get("avg_similarity", ""), tr.get("avg_similarity", ""),
                ts.get("avg_pred_length", ""), tr.get("avg_pred_length", ""),
                urg.get("avg_loss", ""), nonurg.get("avg_loss", ""),
            ]
            f.write("\t".join(str(x) for x in row) + "\n")

    # Save per-type breakdown for best config
    best_path = os.path.join(args.output_dir, "best_config_breakdown.json")
    with open(best_path, "w") as f:
        json.dump(best_result, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"GRID SEARCH COMPLETE — {len(combos)} combinations scored")
    print(f"{'='*70}")
    print(f"\nBest hyperparameters (by test loss):")
    bp = best_result["params"]
    print(f"  alpha={bp['alpha']}  beta={bp['beta']}  gamma={bp['gamma']}")
    print(f"  tau_min={bp['tau_min']}  tau_max={bp['tau_max']}  p={bp['p']}  kappa={bp['kappa']}")
    print(f"\nBest test loss:  {best_result['by_split']['test']['avg_loss']}")
    print(f"Best train loss: {best_result['by_split']['train']['avg_loss']}")
    print(f"\nPer-type breakdown (test, best config):")
    for qtype, stats in sorted(best_result["by_type"].items()):
        test_entries = [e for e in stats] if isinstance(stats, list) else None
        print(f"  {qtype:25s}  loss={stats['avg_loss']:.4f}  sim={stats['avg_similarity']:.4f}  n={stats['n']}")
    print(f"\nUrgent test loss:     {best_result['by_urgency']['urgent'].get('avg_loss', 'N/A')}")
    print(f"Non-urgent test loss: {best_result['by_urgency']['not_urgent'].get('avg_loss', 'N/A')}")
    print(f"\nFiles saved:")
    print(f"  {full_path}  (all {len(combos)} combos)")
    print(f"  {top_path}  (top 20)")
    print(f"  {table_path}  (top 50 as TSV)")
    print(f"  {best_path}  (best config detail)")


if __name__ == "__main__":
    main()
