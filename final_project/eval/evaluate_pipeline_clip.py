"""
evaluate_pipeline_clip.py

Same structure as evaluate_three_conditions.py, but replaces the
zero-shot Kimi-VL classifier (Phase 1) with Julia's trained CLIP classifier.

Phase 1: CLIP classifies all examples (no base model needed)
Phase 2: SFT urgent model — generate for ALL examples  (unchanged)
Phase 3: SFT+DPO non-urgent model — generate for ALL examples  (unchanged)
Phase 4: Route by CLIP predictions, score with composite loss  (unchanged structure)
"""

import csv, json, os, torch
from collections import Counter
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

try:
    import clip
except ImportError:
    raise ImportError(
        "Install CLIP: pip install git+https://github.com/openai/CLIP.git"
    )

# === Paths (same as evaluate_three_conditions.py) ===
BASE_MODEL = "moonshotai/Kimi-VL-A3B-Instruct"
SFT_URGENT_ADAPTER = os.path.expanduser("~/LLaMA-Factory/output/sft_urgent_vision")
SFT_NONURGENT_ADAPTER = os.path.expanduser("~/LLaMA-Factory/output/sft_nonurgent_vision_v2")
DPO_NONURGENT_ADAPTER = os.path.expanduser("~/LLaMA-Factory/output/dpo_nonurgent_sft")
CLIP_MODEL_PATH = os.path.expanduser("~/final_model.pt")
TEST_CSV = os.path.expanduser("~/test_labeled.csv")
BASELINE_PATH = os.path.expanduser("~/baseline_predictions.json")
FRAMES_DIR = os.path.expanduser("~/test")
OUTPUT_PATH = os.path.expanduser("~/FINAL_eval_pipeline_clip.json")

# === System prompts (same as evaluate_three_conditions.py) ===
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

# === Composite loss params (same as evaluate_three_conditions.py) ===
ALPHA = 0.4
BETA = 0.3
GAMMA = 0.3
TAU_MIN = 5
TAU_MAX = 30
P_EXP = 2.0
KAPPA = 0.5


# === Copied verbatim from evaluate_three_conditions.py ===

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


# === NEW: CLIP classifier (replaces Phase 1 zero-shot classification) ===

class CLIPUrgencyClassifier(torch.nn.Module):
    """
    Julia's CLIP urgency classifier.
    Architecture (from state dict inspection):
      - clip_model: CLIP ViT-B/32 (frozen)
      - classifier: Sequential(Linear(1024,512), ReLU, Dropout, Linear(512,1))
    Input: 4 frames mean-pooled + text embedding concatenated.
    """
    def __init__(self):
        super().__init__()
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cpu")
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        # Matches state dict keys: classifier.0, classifier.1, classifier.2, classifier.3
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),   # classifier.0
            torch.nn.ReLU(),               # classifier.1 (no params)
            torch.nn.Dropout(0.2),         # classifier.2 (no params)
            torch.nn.Linear(512, 1),       # classifier.3
        )

    def forward(self, images, text_tokens):
        with torch.no_grad():
            img_feat = self.clip_model.encode_image(images)
            txt_feat = self.clip_model.encode_text(text_tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        img_feat = img_feat.mean(dim=0, keepdim=True).float()
        txt_feat = txt_feat.float()
        combined = torch.cat([img_feat, txt_feat], dim=-1)
        return self.classifier(combined)


def load_clip_classifier(model_path, device):
    """Load Julia's CLIP classifier from nested state dict."""
    print(f"  Loading CLIP classifier from {model_path} "
          f"({os.path.getsize(model_path)/1e6:.0f} MB)...")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # checkpoint is {'model_state': OrderedDict({...})}
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")

    print(f"  State dict has {len(state_dict)} keys")

    # Reconstruct model and load weights
    model = CLIPUrgencyClassifier()
    model.load_state_dict(state_dict, strict=True)
    print(f"  Loaded successfully (strict=True)")

    model = model.to(device)
    model.eval()
    return model


def classify_with_clip(clip_model, clip_preprocess, question_id, question, device,
                       threshold=0.5):
    """Classify one example with Julia's CLIP model. Returns 'urgent' or 'not_urgent'."""
    frame_dir = os.path.join(FRAMES_DIR, question_id)
    if not os.path.isdir(frame_dir):
        return "not_urgent"

    frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    if not frame_files:
        return "not_urgent"

    # Sample up to 4 frames uniformly (matching Julia's training)
    if len(frame_files) >= 4:
        indices = [int(j * (len(frame_files) - 1) / 3) for j in range(4)]
        selected = [frame_files[idx] for idx in indices]
    else:
        selected = frame_files

    frames = [Image.open(os.path.join(frame_dir, f)).convert("RGB") for f in selected]
    frame_tensors = torch.stack([clip_preprocess(f) for f in frames]).to(device)
    text_tokens = clip.tokenize([question], truncate=True).to(device)

    with torch.no_grad():
        # Try direct forward
        try:
            logit = clip_model(frame_tensors, text_tokens)
            if isinstance(logit, tuple):
                logit = logit[0]
            prob = torch.sigmoid(logit.squeeze()).item()
            return "urgent" if prob >= threshold else "not_urgent"
        except Exception:
            pass

        # Fallback: find backbone + head manually
        backbone = head = None
        for name, child in clip_model.named_children():
            if hasattr(child, 'encode_image'):
                backbone = child
            elif isinstance(child, (torch.nn.Sequential, torch.nn.Linear)):
                head = child

        if backbone is None or head is None:
            print(f"  ERROR: Cannot find backbone/head in {type(clip_model).__name__}")
            print(f"  Children: {[n for n, _ in clip_model.named_children()]}")
            return "not_urgent"

        img_feat = backbone.encode_image(frame_tensors)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        img_feat = img_feat.mean(dim=0, keepdim=True)
        txt_feat = backbone.encode_text(text_tokens)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        combined = torch.cat([img_feat.float(), txt_feat.float()], dim=-1)
        logit = head(combined)
        prob = torch.sigmoid(logit.squeeze()).item()
        return "urgent" if prob >= threshold else "not_urgent"


# === Main (same structure as evaluate_three_conditions.py) ===

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("PIPELINE EVALUATION WITH CLIP CLASSIFIER")
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

    # ================================================================
    # PHASE 1: CLIP classification (replaces base model classification)
    # ================================================================
    print("\n=== PHASE 1: CLIP urgency classification ===")
    clip_model = load_clip_classifier(CLIP_MODEL_PATH, device)
    _, clip_preprocess = clip.load("ViT-B/32", device="cpu")

    classifications = {}
    for i, item in enumerate(test_data):
        qid = item["question_id"]
        classifications[qid] = classify_with_clip(
            clip_model, clip_preprocess, qid, item["question"], device
        )
        if (i + 1) % 50 == 0:
            n_pred_urgent = sum(1 for v in classifications.values() if v == "urgent")
            print(f"  [{i+1}/{len(test_data)}] classified {n_pred_urgent} urgent so far")

    del clip_model
    torch.cuda.empty_cache()

    # Print classification stats
    n_pred_urgent = sum(1 for v in classifications.values() if v == "urgent")
    print(f"\n  Classification results: {n_pred_urgent} predicted urgent, "
          f"{len(classifications)-n_pred_urgent} predicted non-urgent")

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

    print(f"\n  CLIP Classifier performance:")
    print(f"    Accuracy:  {correct}/{len(test_data)} = {correct/len(test_data):.3f}")
    print(f"    Precision: {tp}/{tp+fp} = {precision:.3f}")
    print(f"    Recall:    {tp}/{tp+fn} = {recall:.3f}")
    print(f"    F1:        {f1:.3f}")
    print(f"    Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"\n  (vs zero-shot: Acc=0.479 Prec=0.600 Recall=0.004 F1=0.009)")

    # ================================================================
    # PHASE 2: SFT urgent model — generate for ALL examples
    # ================================================================
    print("\n=== PHASE 2: SFT urgent model ===")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
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
    # PHASE 4: Route by CLIP predictions and compute metrics
    # ================================================================
    print("\n=== PHASE 4: Computing metrics ===")

    pipeline_results = []
    for item in test_data:
        qid = item["question_id"]
        is_urgent = item["urgency"] == "urgent"
        pred_urgency = classifications[qid]

        if pred_urgency == "urgent":
            pred = urgent_answers[qid]
        else:
            pred = nonurgent_answers[qid]

        pipeline_results.append({
            "question_id": qid,
            "question": item["question"],
            "gt_urgency": item["urgency"],
            "pred_urgency": pred_urgency,
            "routed_to": "urgent_model" if pred_urgency == "urgent" else "nonurgent_model",
            "prediction": pred,
            "urgent_prediction": urgent_answers[qid],
            "nonurgent_prediction": nonurgent_answers[qid],
            "loss": composite_loss(pred, item["answers"], is_urgent),
            "sim": semantic_similarity(pred, item["answers"]),
            "pred_length": len(pred.split()),
        })

    # Also build oracle results (same models, ground-truth routing)
    oracle_results = []
    for item in test_data:
        qid = item["question_id"]
        is_urgent = item["urgency"] == "urgent"
        pred = urgent_answers[qid] if is_urgent else nonurgent_answers[qid]
        oracle_results.append({
            "question_id": qid,
            "question": item["question"],
            "gt_urgency": item["urgency"],
            "prediction": pred,
            "loss": composite_loss(pred, item["answers"], is_urgent),
            "sim": semantic_similarity(pred, item["answers"]),
            "pred_length": len(pred.split()),
        })

    pipeline_metrics = compute_metrics(pipeline_results)
    oracle_metrics = compute_metrics(oracle_results)

    # Baseline
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

    conditions = {
        "ORACLE": oracle_metrics,
        "CLIP_PIPELINE": pipeline_metrics,
    }
    print_comparison(conditions, baseline_metrics)

    # Classifier impact
    print("\n--- Classifier impact on pipeline ---")
    misrouted = [r for r in pipeline_results if r["pred_urgency"] != r["gt_urgency"]]
    correctly_routed = [r for r in pipeline_results if r["pred_urgency"] == r["gt_urgency"]]
    if correctly_routed:
        print(f"  Correctly routed: n={len(correctly_routed)}  "
              f"loss={sum(r['loss'] for r in correctly_routed)/len(correctly_routed):.4f}  "
              f"sim={sum(r['sim'] for r in correctly_routed)/len(correctly_routed):.4f}")
    if misrouted:
        print(f"  Misrouted:        n={len(misrouted)}  "
              f"loss={sum(r['loss'] for r in misrouted)/len(misrouted):.4f}  "
              f"sim={sum(r['sim'] for r in misrouted)/len(misrouted):.4f}")

    # By question type
    print("\n--- By question type (CLIP PIPELINE) ---")
    qtypes = sorted(set(item["question_type"] for item in test_data))
    for qtype in qtypes:
        qids_of_type = set(item["question_id"] for item in test_data
                           if item["question_type"] == qtype)
        type_results = [r for r in pipeline_results if r["question_id"] in qids_of_type]
        if not type_results:
            continue
        avg_loss = sum(r["loss"] for r in type_results) / len(type_results)
        avg_sim = sum(r["sim"] for r in type_results) / len(type_results)
        print(f"  {qtype:<35} loss={avg_loss:.4f}  sim={avg_sim:.4f}  n={len(type_results)}")

    # Save
    output = {
        "classifier": {
            "type": "clip",
            "model_path": CLIP_MODEL_PATH,
            "accuracy": round(correct / len(test_data), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        },
        "baseline": baseline_metrics,
        "oracle": oracle_metrics,
        "pipeline_clip": pipeline_metrics,
        "comparison_with_zero_shot": {
            "zero_shot_pipeline": {"overall_loss": 0.4807, "overall_sim": 0.3456},
            "clip_pipeline": {
                "overall_loss": pipeline_metrics.get("overall_loss"),
                "overall_sim": pipeline_metrics.get("overall_sim"),
            },
        },
        "details": pipeline_results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {OUTPUT_PATH}")

    summary = {k: v for k, v in output.items() if k != "details"}
    summary_path = os.path.expanduser("~/eval_summary_pipeline_clip.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
