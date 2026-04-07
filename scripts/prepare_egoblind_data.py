"""
prepare_egoblind_data.py (Approach 2: Prompt-Conditioned)

Converts EgoBlind into a SINGLE LLaMA-Factory dataset where each example
has an urgency tag prepended to the system prompt.
- [URGENT] examples use the shortest correct reference answer
- [NON-URGENT] examples use the longest reference answer

Also generates a DPO-ready urgent-only subset for Phase 2.
"""

import argparse
import json
import os
from pathlib import Path


SYSTEM_PROMPT_URGENT = (
    "[URGENT] You are a real-time visual assistant for a blind user. "
    "The user is wearing a camera and asking about their surroundings. "
    "Respond with brief, actionable guidance. Be direct. "
    "If you cannot determine the answer, say so clearly."
)

SYSTEM_PROMPT_NONURGENT = (
    "[NON-URGENT] You are a visual assistant for a blind user. "
    "The user is wearing a camera and asking about their surroundings. "
    "Provide a helpful, detailed answer based on the video content. "
    "If you cannot determine the answer, say so clearly."
)


def select_answer(answers: list[str], mode: str) -> str:
    if not answers:
        return "I don't know."
    if mode == "urgent":
        return min(answers, key=lambda a: len(a.split()))
    else:
        return max(answers, key=lambda a: len(a.split()))


def build_sharegpt_entry(
    question: str,
    answer: str,
    system_prompt: str,
    images: list[str] | None = None,
) -> dict:
    entry = {
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ],
        "system": system_prompt,
    }
    if images:
        entry["images"] = images
    return entry


def extract_frames(video_path: str, output_dir: str, max_frames: int = 8) -> list[str]:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return []
    indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
    indices = [min(idx, total_frames - 1) for idx in indices]
    video_name = Path(video_path).stem
    frame_dir = os.path.join(output_dir, "frames", video_name)
    os.makedirs(frame_dir, exist_ok=True)
    frame_paths = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(frame_dir, f"frame_{i:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
    cap.release()
    return frame_paths


def load_egoblind(egoblind_dir: str, urgency_labels: dict) -> list[dict]:
    """Load EgoBlind and attach urgency labels."""
    # --- ADAPT THIS TO YOUR EGOBLIND FORMAT ---
    ann_path = os.path.join(egoblind_dir, "annotations", "train.json")
    with open(ann_path, "r") as f:
        annotations = json.load(f)

    data = []
    for item in annotations:
        qid = item.get("question_id", item.get("id", ""))
        question = item["question"]
        answers = item.get("answers", item.get("answer", []))
        if isinstance(answers, str):
            answers = [answers]
        video_id = item.get("video_id", item.get("video", ""))
        label = urgency_labels.get(str(qid), "non-urgent")

        data.append({
            "question": question,
            "answers": answers,
            "video_id": video_id,
            "question_id": qid,
            "question_type": item.get("question_type", item.get("type", "")),
            "urgency": label,
        })
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--egoblind_dir", required=True)
    parser.add_argument("--urgency_labels", required=True)
    parser.add_argument("--output_dir", default="data/")
    parser.add_argument("--extract_frames", action="store_true")
    parser.add_argument("--max_frames", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.urgency_labels, "r") as f:
        urgency_labels = json.load(f)

    data = load_egoblind(args.egoblind_dir, urgency_labels)
    n_urgent = sum(1 for d in data if d["urgency"] == "urgent")
    print(f"Loaded {len(data)} examples ({n_urgent} urgent, {len(data)-n_urgent} non-urgent)")

    # -------------------------------------------------------
    # Build COMBINED SFT dataset (single model, tagged)
    # -------------------------------------------------------
    combined_sft = []
    for item in data:
        is_urgent = item["urgency"] == "urgent"
        answer = select_answer(item["answers"], "urgent" if is_urgent else "non-urgent")
        system = SYSTEM_PROMPT_URGENT if is_urgent else SYSTEM_PROMPT_NONURGENT
        images = None
        if args.extract_frames:
            vpath = os.path.join(args.egoblind_dir, "videos", f"{item['video_id']}.mp4")
            if os.path.exists(vpath):
                images = extract_frames(vpath, args.output_dir, args.max_frames)

        entry = build_sharegpt_entry(item["question"], answer, system, images)
        entry["_meta"] = {
            "question_id": item["question_id"],
            "all_answers": item["answers"],
            "urgency": item["urgency"],
        }
        combined_sft.append(entry)

    combined_path = os.path.join(args.output_dir, "egoblind_combined_sft.json")
    with open(combined_path, "w") as f:
        json.dump(combined_sft, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(combined_sft)} combined SFT examples to {combined_path}")

    # -------------------------------------------------------
    # Build URGENT-ONLY subset for DPO (Phase 2)
    # -------------------------------------------------------
    urgent_only = [e for e in combined_sft if e.get("_meta", {}).get("urgency") == "urgent"]
    urgent_path = os.path.join(args.output_dir, "egoblind_urgent_for_dpo.json")
    with open(urgent_path, "w") as f:
        json.dump(urgent_only, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(urgent_only)} urgent examples for DPO to {urgent_path}")

    # -------------------------------------------------------
    # Register in dataset_info.json
    # -------------------------------------------------------
    dataset_info = {
        "egoblind_combined_sft": {
            "file_name": "egoblind_combined_sft.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "images": "images"},
        },
    }

    info_path = os.path.join(args.output_dir, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            existing = json.load(f)
        existing.update(dataset_info)
        dataset_info = existing
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Updated {info_path}")


if __name__ == "__main__":
    main()
