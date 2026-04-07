"""
run_baseline.py

Runs Kimi vision API on all EgoBlind train + test examples.
Extracts frames on-the-fly from local videos and sends them as base64.
Uses urgent/non-urgent prompts based on the urgency column.
Checkpoints every 25 rows.

Usage:
    cd MMAI_Project
    caffeinate -s python3 run_baseline.py --api_key YOUR_KEY
"""

import argparse
import base64
import csv
import cv2
import json
import os
import time
import sys


# ============================================================
# Prompts
# ============================================================

PROMPT_URGENT = (
    "You are a real-time visual assistant for a blind user. "
    "The user is wearing a camera and asking about their surroundings. "
    "These frames are from their egocentric video (start time: {start_time}s). "
    "Respond with brief, actionable guidance. Be direct. "
    "If you cannot determine the answer, say so clearly.\n\n"
    "Question: {question}\n\n"
    "Answer concisely:"
)

PROMPT_NONURGENT = (
    "You are a visual assistant for a blind user. "
    "The user is wearing a camera and asking about their surroundings. "
    "These frames are from their egocentric video (start time: {start_time}s). "
    "Provide a helpful, detailed answer based on the video content. "
    "If you cannot determine the answer, say so clearly.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


# ============================================================
# Frame extraction
# ============================================================

def extract_frames_from_video(video_path, start_time=0.0, num_frames=5):
    """
    Extract frames as base64-encoded JPEGs.
    Samples evenly from start of video up to start_time (real-time QA setup).
    Returns list of base64 strings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(int(start_time * fps), total_frames - 1) if start_time > 0 else total_frames - 1
    if end_frame <= 0:
        end_frame = total_frames - 1

    indices = [int(i * end_frame / num_frames) for i in range(num_frames)]

    frames_b64 = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize to save tokens — 320px wide
            h, w = frame.shape[:2]
            if w > 320:
                scale = 320 / w
                frame = cv2.resize(frame, (320, int(h * scale)))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            frames_b64.append(base64.b64encode(buf).decode("utf-8"))

    cap.release()
    return frames_b64


# ============================================================
# API
# ============================================================

def query_kimi(client, question, start_time, frames_b64, is_urgent, model, max_retries=5):
    template = PROMPT_URGENT if is_urgent else PROMPT_NONURGENT
    prompt = template.format(question=question, start_time=start_time)

    # Build message content: frames first, then text
    content = []
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })
    content.append({"type": "text", "text": prompt})

    for attempt in range(max_retries):
        try:
            start = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=256,
                temperature=0.2,
            )
            elapsed = (time.time() - start) * 1000
            text = response.choices[0].message.content.strip()
            tok_in = response.usage.prompt_tokens if response.usage else 0
            tok_out = response.usage.completion_tokens if response.usage else 0
            return {
                "prediction": text,
                "latency_ms": round(elapsed, 1),
                "tokens_in": tok_in,
                "tokens_out": tok_out,
                "status": "ok",
                "num_frames": len(frames_b64),
            }
        except Exception as e:
            wait = min(2 ** attempt * 2, 60)
            print(f"    API error (attempt {attempt+1}): {e}", flush=True)
            time.sleep(wait)

    return {
        "prediction": "API_ERROR", "latency_ms": 0,
        "tokens_in": 0, "tokens_out": 0, "status": "error", "num_frames": 0,
    }


# ============================================================
# Load CSVs
# ============================================================

def load_csv(path, split_name):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            answers = []
            for i in range(4):
                a = row.get(f"answer{i}", "")
                if a and a.strip():
                    answers.append(a.strip())

            # Skip rows with no answers
            if split_name == "test" and not answers:
                continue

            urgency = row.get("urgency", "not_urgent").strip()
            rows.append({
                "question_id": row.get("question_id", ""),
                "video_name": row.get("video_name", "").strip(),
                "question": row.get("question", ""),
                "answers": answers,
                "question_type": row.get("type", ""),
                "start_time": row.get("start-time/s", "0"),
                "urgency": urgency,
                "split": split_name,
            })
    return rows


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--model", default="moonshot-v1-32k-vision-preview")
    parser.add_argument("--train_csv", default="train_labeled.csv")
    parser.add_argument("--test_csv", default="test_labeled.csv")
    parser.add_argument("--train_videos", default="train_videos")
    parser.add_argument("--test_videos", default="test_videos")
    parser.add_argument("--output", default="baseline_predictions.json")
    parser.add_argument("--checkpoint", default="baseline_checkpoint.json")
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=args.api_key, base_url="https://api.moonshot.ai/v1")

    # Load data
    print("Loading data...", flush=True)
    all_data = []
    if os.path.exists(args.train_csv):
        train = load_csv(args.train_csv, "train")
        print(f"  Train: {len(train)} rows", flush=True)
        all_data.extend(train)
    if os.path.exists(args.test_csv):
        test = load_csv(args.test_csv, "test")
        print(f"  Test: {len(test)} rows", flush=True)
        all_data.extend(test)

    n_urgent = sum(1 for d in all_data if d["urgency"] == "urgent")
    print(f"  Total: {len(all_data)} ({n_urgent} urgent, {len(all_data)-n_urgent} non-urgent)", flush=True)

    # Load checkpoint
    done_ids = set()
    results = []
    if os.path.exists(args.checkpoint):
        with open(args.checkpoint, "r") as f:
            results = json.load(f)
        done_ids = {r["question_id"] for r in results}
        print(f"  Resuming: {len(done_ids)} already done", flush=True)

    remaining = [d for d in all_data if d["question_id"] not in done_ids]
    total_tokens = sum(r.get("tokens_in", 0) + r.get("tokens_out", 0) for r in results)
    errors = sum(1 for r in results if r.get("status") == "error")

    est_minutes = len(remaining) * (args.delay + 2.0) / 60  # vision calls are slower
    print(f"\n  {len(remaining)} calls remaining (~{est_minutes:.0f} minutes)", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Frames per video: {args.num_frames}", flush=True)
    print(f"  Started at: {time.strftime('%H:%M:%S')}", flush=True)
    print(flush=True)

    for i, item in enumerate(remaining):
        is_urgent = item["urgency"] == "urgent"
        start_time = float(item["start_time"] or 0)

        # Find video file
        padded = item["video_name"].zfill(5)
        if item["split"] == "train":
            video_path = os.path.join(args.train_videos, f"{padded}.mp4")
        else:
            video_path = os.path.join(args.test_videos, f"{padded}.mp4")

        # Extract frames on the fly
        frames_b64 = []
        if os.path.exists(video_path):
            frames_b64 = extract_frames_from_video(video_path, start_time, args.num_frames)

        # Query API (sends frames if available, text-only if not)
        api_result = query_kimi(
            client, item["question"], start_time, frames_b64,
            is_urgent, args.model,
        )

        result = {
            "question_id": item["question_id"],
            "video_name": item["video_name"],
            "question": item["question"],
            "answers": item["answers"],
            "question_type": item["question_type"],
            "urgency": item["urgency"],
            "split": item["split"],
            "start_time": start_time,
            "prediction": api_result["prediction"],
            "latency_ms": api_result["latency_ms"],
            "tokens_in": api_result["tokens_in"],
            "tokens_out": api_result["tokens_out"],
            "num_frames": api_result["num_frames"],
            "status": api_result["status"],
        }
        results.append(result)
        total_tokens += api_result["tokens_in"] + api_result["tokens_out"]
        if api_result["status"] == "error":
            errors += 1

        # Checkpoint every 25
        if (i + 1) % 25 == 0 or (i + 1) == len(remaining):
            with open(args.checkpoint, "w") as f:
                json.dump(results, f, ensure_ascii=False)
            pct = (len(done_ids) + i + 1) / len(all_data) * 100
            cost_est = total_tokens / 1_000_000 * 1.5  # rough $/M for vision
            print(
                f"  [{len(done_ids)+i+1}/{len(all_data)}] ({pct:.0f}%) "
                f"tokens={total_tokens} ~${cost_est:.2f} errors={errors} "
                f"time={time.strftime('%H:%M:%S')}",
                flush=True
            )

        time.sleep(args.delay)

    # Save final
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    cost_est = total_tokens / 1_000_000 * 1.5
    print(f"\n{'='*50}", flush=True)
    print(f"DONE — {len(results)} predictions", flush=True)
    print(f"  Train: {sum(1 for r in results if r['split']=='train')}", flush=True)
    print(f"  Test:  {sum(1 for r in results if r['split']=='test')}", flush=True)
    print(f"  Errors: {errors}", flush=True)
    print(f"  Total tokens: {total_tokens} (~${cost_est:.2f})", flush=True)
    print(f"  Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
