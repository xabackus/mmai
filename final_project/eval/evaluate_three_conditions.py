"""
evaluate_three_conditions.py

Three evaluation conditions, all scored with composite loss:

1. BASE MODEL: Kimi-VL with no fine-tuning or routing. Just answers every question.
2. ORACLE ROUTING: Ground-truth urgency labels route to best models
   (SFT urgent + SFT+DPO non-urgent).
3. CLASSIFIER PIPELINE: Base model classifies urgency from frame+question,
   then routes to the appropriate fine-tuned model.

Composite loss: L = α·L_acc + β·L_util + γ·L_lat (urgent) or L = α·L_acc + β·L_util (non-urgent)
γ is applied based on GROUND TRUTH urgency (measures user experience, not system's prediction).
"""

import csv, json, os, torch
from collections import Counter
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

# === Paths ===
BASE_MODEL = "moonshotai/Kimi-VL-A3B-Instruct"
SFT_URGENT_ADAPTER = os.path.expanduser("~/LLaMA-Factory/output/sft_urgent_vision")
SFT_NONURGENT_ADAPTER = os.path.expanduser("~/LLaMA-Factory/output/sft_nonurgent_vision_v2")
DPO_NONURGENT_ADAPTER = os.path.expanduser("~/LLaMA-Factory/output/dpo_nonurgent_sft")
TEST_CSV = os.path.expanduser("~/test_labeled.csv")
BASELINE_PATH = os.path.expanduser("~/baseline_predictions.json")
FRAMES_DIR = os.path.expanduser("~/test")
OUTPUT_PATH = os.path.expanduser("~/eval_three_conditions.json")

# === System prompts ===
SYSTEM_BASE = (
    "You are a visual assistant for a blind user. "
    "The user is wearing a camera and asking about their surroundings. "
    "Provide a helpful answer based on the video content. "
    "If you cannot determine the answer, say so clearly."
)
SYSTEM_URGENT = (
    "You are a real-time visual assistant for a blind user. "
    "The user is wearing a camera and asking about their surroundings. "
    "Respond with brief, actionable guidance. Be direct. "
    "If you cannot determine the answer, say so clearly."
)
SYSTEM_NONURGENT = (
    "You are a visual assistant for a blind user. "
    "The user is wearing a camera and asking about their surroundings. "
    "Provide a helpful, detailed answer based on the video content. "
    "If you cannot determine the answer, say so clearly."
)
SYSTEM_CLASSIFY = (
    "You are a safety triage system for a blind user's visual assistant. "
    "The user is wearing a camera and asking a question about their surroundings. "
    "Based on the image and question, classify this query as either 'urgent' "
    "(immediate safety concern requiring a brief response) or 'not_urgent' "
    "(can receive a detailed answer). Respond with ONLY the word 'urgent' or 'not_urgent'."
)

# === Composite loss params (consistent across all evals) ===
ALPHA = 0.4
BETA = 0.3
GAMMA = 0.3
TAU_MIN = 5
TAU_MAX = 30
P_EXP = 2.0
KAPPA = 0.5


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
        return ALPHA * la + BETA * lu + GAMMA * ll
    else:
        return ALPHA * la + BETA * lu


def load_test_data(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            answers = [row.get(f"answer{i}", "").strip()
                       for i in range(4)
                       if row.get(f"answer{i}", "").strip()]
            if not answers:
                continue
            rows.append({
                "question_id": row.get("question_id", "").strip(),
                "question": row.get("question", "").strip(),
                "answers": answers,
                "question_type": row.get("type", ""),
                "urgency": row.get("urgency", "not_urgent").strip(),
            })
    return rows


def load_frame(question_id):
    frame_dir = os.path.join(FRAMES_DIR, question_id)
    if not os.path.isdir(frame_dir):
        return None
    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    if not frames:
        return None
    img = Image.open(os.path.join(frame_dir, frames[-1])).convert("RGB")
    w, h = img.size
    new_w = 320
    img = img.resize((new_w, int(h * new_w / w)))
    return img


def generate(model, processor, question, system_prompt, image=None, max_new_tokens=128):
    user_content = []
    if image:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": question})

    msgs = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True)

    if image:
        inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
    else:
        inputs = processor(text=text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             temperature=0.2, do_sample=True)
    return processor.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def classify_urgency(model, processor, question, image):
    """Classify query urgency using the base model."""
    response = generate(model, processor, question, SYSTEM_CLASSIFY,
                        image=image, max_new_tokens=10)
    response_lower = response.lower().strip()
    if "not" in response_lower or "non" in response_lower:
        return "not_urgent"
    elif "urgent" in response_lower:
        return "urgent"
    else:
        # Default to not_urgent if unclear
        return "not_urgent"


def load_base_model():
    print("  Loading base model (no adapter)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model


def load_adapter_model(adapter_path):
    print(f"  Loading adapter from {adapter_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def load_stacked_model(sft_path, dpo_path):
    print(f"  Loading base + merging SFT ({sft_path}) + applying DPO ({dpo_path})...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, sft_path)
    model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, dpo_path)
    model.eval()
    return model


def compute_metrics(results_list):
    """Compute aggregate metrics for a list of result dicts."""
    if not results_list:
        return {}

    def avg(key):
        return sum(r[key] for r in results_list) / len(results_list)

    urgent = [r for r in results_list if r["gt_urgency"] == "urgent"]
    nonurgent = [r for r in results_list if r["gt_urgency"] != "urgent"]

    metrics = {
        "overall_loss": round(avg("loss"), 4),
        "overall_sim": round(avg("sim"), 4),
        "overall_len": round(avg("pred_length"), 1),
        "n": len(results_list),
    }
    if urgent:
        metrics["urgent_loss"] = round(sum(r["loss"] for r in urgent) / len(urgent), 4)
        metrics["urgent_sim"] = round(sum(r["sim"] for r in urgent) / len(urgent), 4)
        metrics["urgent_len"] = round(sum(r["pred_length"] for r in urgent) / len(urgent), 1)
        metrics["n_urgent"] = len(urgent)
    if nonurgent:
        metrics["nonurgent_loss"] = round(sum(r["loss"] for r in nonurgent) / len(nonurgent), 4)
        metrics["nonurgent_sim"] = round(sum(r["sim"] for r in nonurgent) / len(nonurgent), 4)
        metrics["nonurgent_len"] = round(sum(r["pred_length"] for r in nonurgent) / len(nonurgent), 1)
        metrics["n_nonurgent"] = len(nonurgent)

    return metrics


def print_comparison(conditions, baseline_metrics=None):
    """Print side-by-side comparison of conditions."""
    print(f"\n{'='*90}")
    header = f"{'METRIC':<25}"
    if baseline_metrics:
        header += f"{'BASELINE':>12}"
    for name in conditions:
        header += f"{name:>15}"
    print(header)
    print(f"{'='*90}")

    rows = [
        ("Overall loss", "overall_loss"),
        ("Overall similarity", "overall_sim"),
        ("Overall pred length", "overall_len"),
        ("", None),
        ("Urgent loss", "urgent_loss"),
        ("Urgent similarity", "urgent_sim"),
        ("Urgent pred length", "urgent_len"),
        ("", None),
        ("Non-urgent loss", "nonurgent_loss"),
        ("Non-urgent similarity", "nonurgent_sim"),
        ("Non-urgent pred length", "nonurgent_len"),
    ]

    for label, key in rows:
        if key is None:
            print()
            continue
        line = f"{label:<25}"
        if baseline_metrics and key in baseline_metrics:
            line += f"{baseline_metrics[key]:>12.4f}"
        elif baseline_metrics:
            line += f"{'—':>12}"
        for name, metrics in conditions.items():
            if key in metrics:
                line += f"{metrics[key]:>15.4f}"
            else:
                line += f"{'—':>15}"
        print(line)

    print(f"{'='*90}")


def main():
    print("=" * 60)
    print("THREE-CONDITION EVALUATION")
    print("=" * 60)

    # Load test data
    print("\n=== Loading test data ===")
    test_data = load_test_data(TEST_CSV)
    n_urgent = sum(1 for d in test_data if d["urgency"] == "urgent")
    n_frames = sum(1 for d in test_data
                   if os.path.isdir(os.path.join(FRAMES_DIR, d["question_id"])))
    print(f"  {len(test_data)} examples ({n_urgent} urgent, "
          f"{len(test_data)-n_urgent} non-urgent)")
    print(f"  {n_frames}/{len(test_data)} have frames")

    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # ================================================================
    # PHASE 1: Base model — generate answers AND classify urgency
    # ================================================================
    print("\n=== PHASE 1: Base model (answers + classification) ===")
    base_model = load_base_model()

    base_answers = {}
    classifications = {}
    for i, item in enumerate(test_data):
        qid = item["question_id"]
        image = load_frame(qid)

        # Generate base model answer
        base_answers[qid] = generate(base_model, processor, item["question"],
                                     SYSTEM_BASE, image=image)

        # Classify urgency
        classifications[qid] = classify_urgency(base_model, processor,
                                                item["question"], image)

        if (i + 1) % 50 == 0:
            n_pred_urgent = sum(1 for v in classifications.values() if v == "urgent")
            print(f"  [{i+1}/{len(test_data)}] classified {n_pred_urgent} urgent so far")

    del base_model
    torch.cuda.empty_cache()

    # Print classification stats
    n_pred_urgent = sum(1 for v in classifications.values() if v == "urgent")
    print(f"\n  Classification results: {n_pred_urgent} predicted urgent, "
          f"{len(classifications)-n_pred_urgent} predicted non-urgent")

    # Classification accuracy
    correct = sum(1 for item in test_data
                  if classifications[item["question_id"]] == item["urgency"])
    gt_urgent = [item for item in test_data if item["urgency"] == "urgent"]
    gt_nonurgent = [item for item in test_data if item["urgency"] != "urgent"]
    tp = sum(1 for item in gt_urgent if classifications[item["question_id"]] == "urgent")
    fp = sum(1 for item in gt_nonurgent if classifications[item["question_id"]] == "urgent")
    fn = sum(1 for item in gt_urgent if classifications[item["question_id"]] != "urgent")
    tn = sum(1 for item in gt_nonurgent if classifications[item["question_id"]] != "urgent")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  Classifier performance:")
    print(f"    Accuracy:  {correct}/{len(test_data)} = {correct/len(test_data):.3f}")
    print(f"    Precision: {tp}/{tp+fp} = {precision:.3f}")
    print(f"    Recall:    {tp}/{tp+fn} = {recall:.3f}")
    print(f"    F1:        {f1:.3f}")
    print(f"    Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

    # ================================================================
    # PHASE 2: SFT urgent model — generate for ALL examples
    # ================================================================
    print("\n=== PHASE 2: SFT urgent model ===")
    urgent_model = load_adapter_model(SFT_URGENT_ADAPTER)

    urgent_answers = {}
    for i, item in enumerate(test_data):
        qid = item["question_id"]
        image = load_frame(qid)
        urgent_answers[qid] = generate(urgent_model, processor, item["question"],
                                       SYSTEM_URGENT, image=image)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(test_data)}]")

    del urgent_model
    torch.cuda.empty_cache()

    # ================================================================
    # PHASE 3: SFT+DPO non-urgent model — generate for ALL examples
    # ================================================================
    print("\n=== PHASE 3: SFT+DPO non-urgent model ===")
    nonurgent_model = load_stacked_model(SFT_NONURGENT_ADAPTER, DPO_NONURGENT_ADAPTER)

    nonurgent_answers = {}
    for i, item in enumerate(test_data):
        qid = item["question_id"]
        image = load_frame(qid)
        nonurgent_answers[qid] = generate(nonurgent_model, processor, item["question"],
                                          SYSTEM_NONURGENT, image=image)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(test_data)}]")

    del nonurgent_model
    torch.cuda.empty_cache()

    # ================================================================
    # PHASE 4: Construct conditions and compute metrics
    # ================================================================
    print("\n=== PHASE 4: Computing metrics ===")

    cond1_results = []  # Base model
    cond2_results = []  # Oracle routing
    cond3_results = []  # Classifier pipeline

    for item in test_data:
        qid = item["question_id"]
        is_urgent = item["urgency"] == "urgent"

        # Condition 1: Base model
        pred1 = base_answers[qid]
        cond1_results.append({
            "question_id": qid,
            "question": item["question"],
            "gt_urgency": item["urgency"],
            "prediction": pred1,
            "loss": composite_loss(pred1, item["answers"], is_urgent),
            "sim": semantic_similarity(pred1, item["answers"]),
            "pred_length": len(pred1.split()),
        })

        # Condition 2: Oracle routing (ground truth urgency)
        if is_urgent:
            pred2 = urgent_answers[qid]
        else:
            pred2 = nonurgent_answers[qid]
        cond2_results.append({
            "question_id": qid,
            "question": item["question"],
            "gt_urgency": item["urgency"],
            "prediction": pred2,
            "loss": composite_loss(pred2, item["answers"], is_urgent),
            "sim": semantic_similarity(pred2, item["answers"]),
            "pred_length": len(pred2.split()),
        })

        # Condition 3: Classifier pipeline (predicted urgency for routing,
        # ground truth urgency for loss computation)
        pred_urgency = classifications[qid]
        if pred_urgency == "urgent":
            pred3 = urgent_answers[qid]
        else:
            pred3 = nonurgent_answers[qid]
        cond3_results.append({
            "question_id": qid,
            "question": item["question"],
            "gt_urgency": item["urgency"],
            "pred_urgency": pred_urgency,
            "routed_to": "urgent_model" if pred_urgency == "urgent" else "nonurgent_model",
            "prediction": pred3,
            "loss": composite_loss(pred3, item["answers"], is_urgent),
            "sim": semantic_similarity(pred3, item["answers"]),
            "pred_length": len(pred3.split()),
        })

    # Compute aggregate metrics
    cond1_metrics = compute_metrics(cond1_results)
    cond2_metrics = compute_metrics(cond2_results)
    cond3_metrics = compute_metrics(cond3_results)

    # Compute baseline metrics
    baseline_metrics = None
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
        bl_results = []
        for bp in baseline:
            if bp.get("split") != "test" or not bp.get("answers"):
                continue
            is_urg = bp["urgency"] == "urgent"
            bl_results.append({
                "gt_urgency": bp["urgency"],
                "loss": composite_loss(bp["prediction"], bp["answers"], is_urg),
                "sim": semantic_similarity(bp["prediction"], bp["answers"]),
                "pred_length": len(bp["prediction"].split()),
            })
        baseline_metrics = compute_metrics(bl_results)

    # Print results
    conditions = {
        "BASE": cond1_metrics,
        "ORACLE": cond2_metrics,
        "PIPELINE": cond3_metrics,
    }
    print_comparison(conditions, baseline_metrics)

    # Print classifier impact
    print("\n--- Classifier impact on pipeline ---")
    misrouted = [r for r in cond3_results if r["pred_urgency"] != r["gt_urgency"]]
    correctly_routed = [r for r in cond3_results if r["pred_urgency"] == r["gt_urgency"]]
    if correctly_routed:
        print(f"  Correctly routed: n={len(correctly_routed)}  "
              f"loss={sum(r['loss'] for r in correctly_routed)/len(correctly_routed):.4f}  "
              f"sim={sum(r['sim'] for r in correctly_routed)/len(correctly_routed):.4f}")
    if misrouted:
        print(f"  Misrouted:        n={len(misrouted)}  "
              f"loss={sum(r['loss'] for r in misrouted)/len(misrouted):.4f}  "
              f"sim={sum(r['sim'] for r in misrouted)/len(misrouted):.4f}")

    # By question type
    print("\n--- By question type (ORACLE condition) ---")
    for qtype in sorted(set(r["question"] for r in cond2_results)):
        pass  # Skip — use question_type from test_data instead

    qtypes = sorted(set(item["question_type"] for item in test_data))
    for qtype in qtypes:
        qids_of_type = set(item["question_id"] for item in test_data
                           if item["question_type"] == qtype)
        type_results = [r for r in cond2_results if r["question_id"] in qids_of_type]
        if not type_results:
            continue
        avg_loss = sum(r["loss"] for r in type_results) / len(type_results)
        avg_sim = sum(r["sim"] for r in type_results) / len(type_results)
        print(f"  {qtype:<35} loss={avg_loss:.4f}  sim={avg_sim:.4f}  n={len(type_results)}")

    # Save everything
    output = {
        "classifier": {
            "accuracy": round(correct / len(test_data), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        },
        "baseline": baseline_metrics,
        "condition1_base": cond1_metrics,
        "condition2_oracle": cond2_metrics,
        "condition3_pipeline": cond3_metrics,
        "details": {
            "condition1": cond1_results,
            "condition2": cond2_results,
            "condition3": cond3_results,
        },
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {OUTPUT_PATH}")

    # Summary file
    summary = {k: v for k, v in output.items() if k != "details"}
    summary_path = os.path.expanduser("~/eval_summary_three_conditions.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
