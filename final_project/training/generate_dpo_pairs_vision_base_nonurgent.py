"""
generate_dpo_pairs_vision_base_nonurgent.py

Generate DPO training pairs from the BASE Kimi-VL model for non-urgent queries.
Uses vision input. Scores against longest reference.
Checkpoints every 50 examples, auto-resumable.
"""

import csv
import json
import os
import time
from collections import Counter

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

BASE_MODEL = "moonshotai/Kimi-VL-A3B-Instruct"
TRAIN_CSV = os.path.expanduser("~/train_labeled.csv")
FRAMES_DIR = os.path.expanduser("~/train")
OUTPUT_PATH = os.path.expanduser("~/LLaMA-Factory/data/egoblind_nonurgent_dpo_base.json")
CHECKPOINT_PATH = os.path.expanduser("~/dpo_base_nonurgent_checkpoint.json")
DATASET_INFO_PATH = os.path.expanduser("~/LLaMA-Factory/data/dataset_info.json")

NUM_CANDIDATES = 4
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.9
TOP_P = 0.95

SYSTEM_NONURGENT = (
    "You are a visual assistant for a blind user. "
    "The user is wearing a camera and asking about their surroundings. "
    "Provide a helpful, detailed answer based on the video content. "
    "If you cannot determine the answer, say so clearly."
)


def token_overlap_f1(prediction, reference):
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
    if not references:
        return 0.0
    longest_ref = max(references, key=lambda r: len(r.split()))
    return token_overlap_f1(candidate, longest_ref)


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


def get_frame_path(question_id):
    frame_dir = os.path.join(FRAMES_DIR, question_id)
    if not os.path.isdir(frame_dir):
        return None
    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    if not frames:
        return None
    return os.path.join(frame_dir, frames[-1])


def load_training_data():
    rows = []
    with open(TRAIN_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("urgency", "").strip() == "urgent":
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


def main():
    print("Loading training data...")
    train_data = load_training_data()
    print(f"Found {len(train_data)} non-urgent training examples with frames")

    completed = {}
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            completed = json.load(f)
        print(f"Resuming from checkpoint: {len(completed)} examples already done")

    print(f"Loading base model from {BASE_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    dpo_data = [v for v in completed.values() if v is not None]
    start_time = time.time()

    for idx, item in enumerate(train_data):
        key = str(idx)
        if key in completed:
            continue

        qid = item["question_id"]
        question = item["question"]
        references = item["answers"]

        image = load_frame(qid)
        frame_path = get_frame_path(qid)

        user_content = []
        if image is not None:
            user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": question})

        msgs = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_NONURGENT}]},
            {"role": "user", "content": user_content},
        ]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True)

        if image is not None:
            inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
        else:
            inputs = processor(text=text, return_tensors="pt").to(model.device)

        candidates = []
        for _ in range(NUM_CANDIDATES):
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

        scored = [(c, score_candidate(c, references)) for c in candidates]
        scored.sort(key=lambda x: -x[1])
        chosen = scored[0][0]
        rejected = scored[-1][0]

        if chosen.strip() == rejected.strip():
            completed[key] = None
            continue

        entry = {
            "conversations": [
                {"from": "system", "value": SYSTEM_NONURGENT},
                {"from": "human", "value": "<image>\n" + question},
            ],
            "chosen": {"from": "gpt", "value": chosen},
            "rejected": {"from": "gpt", "value": rejected},
            "images": [frame_path] if frame_path else [],
        }
        dpo_data.append(entry)
        completed[key] = entry

        done = len(completed)
        if done % 10 == 0:
            elapsed = time.time() - start_time
            eta = (len(train_data) - done) * elapsed / max(done, 1) / 3600
            n_valid = len(dpo_data)
            print(f"  [{done}/{len(train_data)}] valid={n_valid} "
                  f"chosen={scored[0][1]:.3f} rejected={scored[-1][1]:.3f} "
                  f"ETA={eta:.1f}h")

        if done % 50 == 0:
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(completed, f)
            print(f"  Checkpoint saved ({done} done)")

    dpo_data = [d for d in dpo_data if d is not None]
    with open(OUTPUT_PATH, "w") as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)
    print(f"\nDone! {len(dpo_data)} DPO pairs saved to {OUTPUT_PATH}")

    if os.path.exists(DATASET_INFO_PATH):
        with open(DATASET_INFO_PATH) as f:
            info = json.load(f)
    else:
        info = {}

    info["egoblind_nonurgent_dpo_base"] = {
        "file_name": "egoblind_nonurgent_dpo_base.json",
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

    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(completed, f)


if __name__ == "__main__":
    main()
