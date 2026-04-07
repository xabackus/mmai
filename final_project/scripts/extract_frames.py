"""
extract_frames.py — Run on your laptop before your flight.

Extracts 5 frames per video as compressed JPEGs.
~450MB total output vs 67GB of raw video.

Usage:
    cd MMAI_Project
    python3 extract_frames.py
"""

import cv2
import os
import csv
import sys


TRAIN_CSV = "train_labeled.csv"
TEST_CSV = "test_labeled.csv"
TRAIN_VIDEO_DIR = "train_videos"
TEST_VIDEO_DIR = "test_videos"
OUTPUT_DIR = "frames"
NUM_FRAMES = 5
JPEG_QUALITY = 70  # lower = smaller files


def extract_frames(video_path, start_time, output_dir, video_name, num_frames=5):
    """Extract num_frames evenly spaced frames from video, starting at start_time."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_time * fps)
    # Use frames from 0 to start_frame (matching EgoBlind's real-time QA setup)
    end_frame = min(start_frame, total_frames - 1) if start_frame > 0 else total_frames - 1

    if end_frame <= 0:
        end_frame = total_frames - 1

    indices = [int(i * end_frame / num_frames) for i in range(num_frames)]

    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            path = os.path.join(output_dir, f"{i}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved += 1
    cap.release()
    return saved


def process_split(csv_path, video_dir, split_name):
    """Process all videos in a split."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Collect unique video + start_time combos
    seen = set()
    tasks = []
    for row in rows:
        vname = row["video_name"].strip()
        padded = vname.zfill(5)
        start_time = float(row.get("start-time/s", 0) or 0)
        qid = row.get("question_id", "")

        key = (padded, start_time, qid)
        if key in seen:
            continue
        seen.add(key)

        video_path = os.path.join(video_dir, f"{padded}.mp4")
        frame_dir = os.path.join(OUTPUT_DIR, split_name, f"{padded}_{qid}")
        tasks.append((video_path, start_time, frame_dir, padded))

    print(f"\n{split_name}: {len(tasks)} unique video-question pairs")
    done = 0
    skipped = 0

    for i, (vpath, start_time, fdir, vname) in enumerate(tasks):
        if os.path.exists(fdir) and len(os.listdir(fdir)) == NUM_FRAMES:
            done += 1
            continue

        if not os.path.exists(vpath):
            skipped += 1
            continue

        n = extract_frames(vpath, start_time, fdir, vname, NUM_FRAMES)
        done += 1

        if (done + skipped) % 200 == 0:
            print(f"  [{done + skipped}/{len(tasks)}] extracted={done} skipped={skipped}")

    print(f"  Done: {done} extracted, {skipped} skipped")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(TRAIN_CSV):
        process_split(TRAIN_CSV, TRAIN_VIDEO_DIR, "train")
    else:
        print(f"WARNING: {TRAIN_CSV} not found, skipping train")

    if os.path.exists(TEST_CSV):
        process_split(TEST_CSV, TEST_VIDEO_DIR, "test")
    else:
        print(f"WARNING: {TEST_CSV} not found, skipping test")

    # Check final size
    total_size = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))

    print(f"\nTotal frames size: {total_size / 1024 / 1024:.0f} MB")
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"\nNext: scp -r frames/ xabackus@athena.dialup.mit.edu:~/egoblind-ra-baseline/")


if __name__ == "__main__":
    main()
