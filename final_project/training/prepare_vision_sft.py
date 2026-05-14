"""
prepare_vision_sft.py

Builds two LLaMA-Factory SFT datasets (urgent + non-urgent) that include
image paths for vision-conditioned training.

Each example uses the last frame from the training frame directory.
Answer selection: shortest for urgent, longest for non-urgent.

Run on login node (no GPU needed):
    python ~/prepare_vision_sft.py
"""

import csv
import json
import os

TRAIN_CSV = os.path.expanduser("~/train_labeled.csv")
FRAMES_DIR = os.path.expanduser("~/train")
OUTPUT_DIR = os.path.expanduser("~/LLaMA-Factory/data")

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


def get_last_frame(question_id):
    """Return absolute path to last frame for this question, or None."""
    frame_dir = os.path.join(FRAMES_DIR, question_id)
    if not os.path.isdir(frame_dir):
        return None
    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    if not frames:
        return None
    return os.path.join(frame_dir, frames[-1])


def main():
    # Read training data
    with open(TRAIN_CSV) as f:
        rows = list(csv.DictReader(f))

    urgent_data = []
    nonurgent_data = []
    n_with_frames = 0
    n_without_frames = 0
    n_no_answers = 0

    for row in rows:
        answers = [row.get(f"answer{i}", "").strip()
                   for i in range(4)
                   if row.get(f"answer{i}", "").strip()]
        if not answers:
            n_no_answers += 1
            continue

        qid = row.get("question_id", "").strip()
        question = row.get("question", "").strip()
        urgency = row.get("urgency", "not_urgent").strip()

        # Get frame path
        frame_path = get_last_frame(qid)
        if frame_path is None:
            n_without_frames += 1
            continue
        n_with_frames += 1

        # Select target answer: shortest for urgent, longest for non-urgent
        if urgency == "urgent":
            answer = min(answers, key=lambda a: len(a.split()))
        else:
            answer = max(answers, key=lambda a: len(a.split()))

        # Build LLaMA-Factory sharegpt entry with image
        # <image> tag tells LLaMA-Factory where to place image tokens
        entry = {
            "conversations": [
                {"from": "human", "value": "<image>\n" + question},
                {"from": "gpt", "value": answer},
            ],
            "system": SYSTEM_URGENT if urgency == "urgent" else SYSTEM_NONURGENT,
            "images": [frame_path],
        }

        if urgency == "urgent":
            urgent_data.append(entry)
        else:
            nonurgent_data.append(entry)

    # Print stats
    print(f"Total CSV rows:      {len(rows)}")
    print(f"Rows without answers: {n_no_answers}")
    print(f"With frames:         {n_with_frames}")
    print(f"Without frames:      {n_without_frames}")
    print(f"Urgent examples:     {len(urgent_data)}")
    print(f"Non-urgent examples: {len(nonurgent_data)}")

    # Show answer length distributions
    if urgent_data:
        u_lens = [len(e["conversations"][1]["value"].split()) for e in urgent_data]
        print(f"\nUrgent answer lengths: min={min(u_lens)} "
              f"avg={sum(u_lens)/len(u_lens):.1f} max={max(u_lens)}")
    if nonurgent_data:
        nu_lens = [len(e["conversations"][1]["value"].split()) for e in nonurgent_data]
        print(f"Non-urgent answer lengths: min={min(nu_lens)} "
              f"avg={sum(nu_lens)/len(nu_lens):.1f} max={max(nu_lens)}")

    # Show samples
    if urgent_data:
        e = urgent_data[0]
        print(f"\nSample urgent:")
        print(f"  Human: {e['conversations'][0]['value'][:80]}")
        print(f"  GPT:   {e['conversations'][1]['value']}")
        print(f"  Image: {e['images'][0]}")
    if nonurgent_data:
        e = nonurgent_data[0]
        print(f"\nSample non-urgent:")
        print(f"  Human: {e['conversations'][0]['value'][:80]}")
        print(f"  GPT:   {e['conversations'][1]['value'][:80]}")
        print(f"  Image: {e['images'][0]}")

    # Save datasets
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    urgent_path = os.path.join(OUTPUT_DIR, "egoblind_urgent_vision_sft.json")
    with open(urgent_path, "w") as f:
        json.dump(urgent_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved urgent dataset to {urgent_path}")

    nonurgent_path = os.path.join(OUTPUT_DIR, "egoblind_nonurgent_vision_sft.json")
    with open(nonurgent_path, "w") as f:
        json.dump(nonurgent_data, f, indent=2, ensure_ascii=False)
    print(f"Saved non-urgent dataset to {nonurgent_path}")

    # Register in dataset_info.json
    info_path = os.path.join(OUTPUT_DIR, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    else:
        info = {}

    for name in ["egoblind_urgent_vision_sft", "egoblind_nonurgent_vision_sft"]:
        info[name] = {
            "file_name": f"{name}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "images": "images",
            },
        }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Registered datasets in {info_path}")


if __name__ == "__main__":
    main()
