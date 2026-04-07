"""
save_frames.py

Re-extracts the same frames used during the baseline run and saves them to disk.
Same parameters: 5 frames, 320px wide, JPEG quality 60.

Usage:
    cd MMAI_Project
    python3 save_frames.py
"""

import csv
import cv2
import json
import os
import sys


def extract_and_save(video_path, start_time, output_dir, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(int(start_time * fps), total_frames - 1) if start_time > 0 else total_frames - 1
    if end_frame <= 0:
        end_frame = total_frames - 1

    indices = [int(i * end_frame / num_frames) for i in range(num_frames)]

    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            if w > 320:
                scale = 320 / w
                frame = cv2.resize(frame, (320, int(h * scale)))
            path = os.path.join(output_dir, f"frame_{i}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            saved += 1

    cap.release()
    return saved


def main():
    predictions_path = "baseline_predictions.json"
    output_root = "baseline_frames"

    with open(predictions_path, "r") as f:
        predictions = json.load(f)

    print(f"Extracting frames for {len(predictions)} predictions...")
    print(f"Output: {output_root}/")

    done = 0
    skipped = 0

    for i, pred in enumerate(predictions):
        qid = pred["question_id"]
        vname = pred["video_name"].strip().zfill(5)
        split = pred["split"]
        start_time = float(pred.get("start_time", 0) or 0)

        if split == "train":
            video_path = os.path.join("train_videos", f"{vname}.mp4")
        else:
            video_path = os.path.join("test_videos", f"{vname}.mp4")

        frame_dir = os.path.join(output_root, split, qid)

        # Skip if already extracted
        if os.path.exists(frame_dir) and len(os.listdir(frame_dir)) == 5:
            done += 1
            continue

        if not os.path.exists(video_path):
            skipped += 1
            continue

        n = extract_and_save(video_path, start_time, frame_dir)
        done += 1

        if (i + 1) % 500 == 0:
            size_mb = sum(
                os.path.getsize(os.path.join(r, f))
                for r, _, files in os.walk(output_root)
                for f in files
            ) / 1024 / 1024
            print(f"  [{i+1}/{len(predictions)}] done={done} skipped={skipped} size={size_mb:.0f}MB")

    # Final size
    total_size = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(output_root)
        for f in files
    ) / 1024 / 1024

    print(f"\nDone: {done} extracted, {skipped} skipped")
    print(f"Total size: {total_size:.0f} MB")
    print(f"Output: {output_root}/")


if __name__ == "__main__":
    main()
