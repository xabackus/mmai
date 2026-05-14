"""
generate_dpo_pairs_vision.py

Generate DPO training pairs for the urgent model using vision.

Pipeline:
1. Load base model + merge SFT urgent vision adapter
2. For each urgent training example:
   - Load the training frame
   - Generate 4 candidates with vision input
   - Score each against the LONGEST reference (incentivizes informativeness over bare yes/no)
   - Pick lowest-loss as chosen, highest-loss as rejected
3. Save in LLaMA-Factory DPO sharegpt format with image paths

Run in egoblind-eval conda env.
"""

import json
import os
import time
from collections import Counter

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

# === Config ===
BASE_MODEL = "moonshotai/Kimi-VL-A3B-Instruct"
SFT_ADAPTER = os.path.expanduser("~/LLaMA-Factory/output/sft_urgent_vision")
TRAIN_CSV = os.path.expanduser("~/train_labeled.csv")
FRAMES_DIR = os.path.expanduser("~/train")
OUTPUT_PATH = os.path.expanduser("~/LLaMA-Factory/data/egoblind_urgent_dpo_vision.json")
CHECKPOINT_PATH = os.path.expanduser("~/dpo_vision_checkpoint.json")
DATASET_INFO_PATH = os.path.expanduser("~/LLaMA-Factory/data/dataset_info.json")

NUM_CANDIDATES = 4
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.9
TOP_P = 0.95

SYSTEM_URGENT = (
    "You are a real-time visual assistant for a blind user. "
    "The user is wearing a camera and asking about their surroundings. "
    "Respond with brief, actionable guidance. Be direct. "
    "If you cannot determine the answer, say so clearly."
)


def token_overlap_f1(prediction, reference):
    """Token-overlap F1 between prediction and reference."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_set = Counter(pred_tokens)
    ref_set = Counter(ref_tokens)
    overlap = sum((pred_set & ref_set).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def score_candidate(candidate, references):
    """
    Score against the LONGEST reference only.
    This incentivizes informative answers over bare yes/no,
    because "Yes" has low F1 against "Yes, there are roadblocks ahead."
    """
    if not references:
        return 0.0
    longest_ref = max(references, key=lambda r: len(r.split()))
    return token_overlap_f1(candidate, longest_ref)


def load_frame(question_id):
    """Load last frame, resized to 320px wide."""
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


def get_frame_path(question_id):
    """Get absolute path to last frame."""
    frame_dir = os.path.join(FRAMES_DIR, question_id)
    if not os.path.isdir(frame_dir):
        return None
    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    if not frames:
        return None
    return os.path.join(frame_dir, frames[-1])


def load_training_data():
    """Load urgent training examples that have frames."""
    import csv
    rows = []
    with open(TRAIN_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("urgency", "").strip() != "urgent":
                continue
            answers = [row.get(f"answer{i}", "").strip()
                       for i in range(4)
                       if row.get(f"answer{i}", "").strip()]
            if not answers:
                continue
            qid = row.get("question_id", "").strip()
            if not os.path.isdir(os.path.join(FRAMES_DIR, qid)):
                continue
            rows.append({
                "question_id": qid,
                "question": row.get("question", "").strip(),
                "answers": answers,
            })
    return rows


def load_model():
    """Load base model with SFT adapter merged."""
    print(f"Loading base model from {BASE_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    print(f"Loading and merging SFT adapter from {SFT_ADAPTER}...")
    model = PeftModel.from_pretrained(model, SFT_ADAPTER)
    model = model.merge_and_unload()
    model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    return model, processor


def generate_candidates(model, processor, question, image, n=NUM_CANDIDATES):
    """Generate n candidate responses with vision input."""
    user_content = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": question})

    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_URGENT}]},
        {"role": "user", "content": user_content},
    ]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True)

    if image is not None:
        inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
    else:
        inputs = processor(text=text, return_tensors="pt").to(model.device)

    candidates = []
    for _ in range(n):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
            )
        gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        candidate = processor.decode(gen_ids, skip_special_tokens=True).strip()
        candidates.append(candidate)

    return candidates


def main():
    # Load training data
    print("Loading training data...")
    train_data = load_training_data()
    print(f"Found {len(train_data)} urgent training examples with frames")

    # Load checkpoint if exists
    completed = {}
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            completed = json.load(f)
        print(f"Resuming from checkpoint: {len(completed)} examples already done")

    # Load model
    model, processor = load_model()

    # Generate pairs
    dpo_data = [v for v in completed.values() if v is not None]
    start_time = time.time()
    n_skipped_start = len([v for v in completed.values() if v is None])

    for idx, item in enumerate(train_data):
        key = str(idx)
        if key in completed:
            continue

        qid = item["question_id"]
        question = item["question"]
        references = item["answers"]

        # Load frame
        image = load_frame(qid)
        frame_path = get_frame_path(qid)

        # Generate candidates
        candidates = generate_candidates(model, processor, question, image)

        # Score each against longest reference
        scored = []
        for c in candidates:
            s = score_candidate(c, references)
            scored.append((c, s))

        scored.sort(key=lambda x: -x[1])  # highest score first
        chosen = scored[0][0]    # best match
        rejected = scored[-1][0]  # worst match

        # Skip if identical
        if chosen.strip() == rejected.strip():
            completed[key] = None
            n_skipped_start += 1
            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(train_data)}] Skipped (identical candidates)")
            continue

        entry = {
            "conversations": [
                {
                    "from": "system",
                    "value": SYSTEM_URGENT,
                },
                {
                    "from": "human",
                    "value": "<image>\n" + question,
                },
            ],
            "chosen": {"from": "gpt", "value": chosen},
            "rejected": {"from": "gpt", "value": rejected},
            "images": [frame_path] if frame_path else [],
        }
        dpo_data.append(entry)
        completed[key] = entry

        # Progress
        done = len(completed)
        n_valid = len(dpo_data)
        elapsed = time.time() - start_time
        if done % 10 == 0 and done > 0:
            rate = elapsed / max(done - len([v for v in completed.values() if v is None and str(list(completed.keys()).index(next(k for k,v2 in completed.items() if v2 is None))) not in completed]), 1)
            eta = (len(train_data) - done) * elapsed / max(done, 1) / 3600
            print(
                f"  [{done}/{len(train_data)}] valid_pairs={n_valid} "
                f"chosen_score={scored[0][1]:.3f} "
                f"rejected_score={scored[-1][1]:.3f} "
                f"ETA={eta:.1f}h"
            )

        # Checkpoint every 50
        if done % 50 == 0:
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(completed, f)
            print(f"  Checkpoint saved ({done} done, {n_valid} valid pairs)")

    # Final save
    dpo_data = [d for d in dpo_data if d is not None]
    with open(OUTPUT_PATH, "w") as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)
    print(f"\nDone! {len(dpo_data)} DPO pairs saved to {OUTPUT_PATH}")

    # Register dataset
    if os.path.exists(DATASET_INFO_PATH):
        with open(DATASET_INFO_PATH) as f:
            info = json.load(f)
    else:
        info = {}

    info["egoblind_urgent_dpo_vision"] = {
        "file_name": "egoblind_urgent_dpo_vision.json",
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
            "images": "images",
        },
    }
    with open(DATASET_INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Registered dataset in {DATASET_INFO_PATH}")

    # Final checkpoint
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(completed, f)


if __name__ == "__main__":
    main()
