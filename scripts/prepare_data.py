"""
Process train and test CSVs into JSONL files + extract frames from videos.

Run from inside your MMAI_Project folder:
    python prepare_data.py

Expected folder structure:
    MMAI_Project/
    ├── train_labeled.csv
    ├── test_labeled.csv
    ├── train_videos/
    │   ├── 00000.mp4
    │   ├── 00001.mp4
    │   └── ...
    ├── test_videos/
    │   ├── 00500.mp4   (or whatever)
    │   └── ...
    └── prepare_data.py   <-- this script

Output:
    MMAI_Project/
    ├── mmai-train/
    │   ├── images/
    │   │   ├── v_00000_1.jpg
    │   │   └── ...
    │   └── data.jsonl
    └── mmai-test/
        ├── images/
        │   ├── v_00500_1.jpg
        │   └── ...
        └── data.jsonl
"""

import pandas as pd
import cv2
import json
import os

# ============================================================
# ######################## CHANGE ME #########################
# ============================================================

# Set these to match your folder structure
TRAIN_CSV = "train_labeled.csv"
TEST_CSV = "test_labeled.csv"
TRAIN_VIDEO_DIR = "train_videos"
TEST_VIDEO_DIR = "test_videos"

# Video file extension (change if your videos are .avi, .mov, etc.)
VIDEO_EXT = ".mp4"

# ============================================================
# ###################### END CHANGE ME #######################
# ============================================================


def process_split(csv_path, video_dir, output_dir, skip_missing_answers=False):
    """
    Reads CSV, extracts frames at the given timestamps, writes JSONL.
    
    If skip_missing_answers=True, rows where answer0 is empty/NaN are skipped.
    Does NOT modify the source CSV in any way.
    """
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    records = []
    skipped = 0
    failed_frames = 0
    missing_videos = 0

    for _, row in df.iterrows():
        # Skip rows with no answer if requested
        if skip_missing_answers:
            if pd.isna(row.get("answer0")) or str(row.get("answer0", "")).strip() == "":
                skipped += 1
                continue

        question_id = row["question_id"]
        video_name = str(row["video_name"]).zfill(5)  # zero-pad to 5 digits
        video_path = os.path.join(video_dir, video_name + VIDEO_EXT)

        if not os.path.exists(video_path):
            missing_videos += 1
            if missing_videos <= 5:
                print(f"  Missing video: {video_path}")
            elif missing_videos == 6:
                print(f"  ... (suppressing further missing video warnings)")
            continue

        # Extract frame at the given timestamp
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print(f"  Could not read FPS for {video_path}")
            cap.release()
            failed_frames += 1
            continue

        target_frame = int(float(row["start-time/s"]) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            failed_frames += 1
            continue

        # Save frame
        img_filename = f"{question_id}.jpg"
        img_path = os.path.join(output_dir, "images", img_filename)
        cv2.imwrite(img_path, frame)

        # Build JSONL entry
        entry = {
            "image": f"images/{img_filename}",
            "question": row["question"],
            "answer": str(row["answer0"]) if not pd.isna(row.get("answer0")) else ""
        }
        records.append(entry)

    # Write JSONL
    jsonl_path = os.path.join(output_dir, "data.jsonl")
    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"  Wrote {len(records)} entries to {jsonl_path}")
    if skipped > 0:
        print(f"  Skipped {skipped} rows (missing answers)")
    if missing_videos > 0:
        print(f"  Missing videos: {missing_videos}")
    if failed_frames > 0:
        print(f"  Failed frame extractions: {failed_frames}")

    return records


if __name__ == "__main__":
    print("Processing TRAIN split...")
    train_records = process_split(
        csv_path=TRAIN_CSV,
        video_dir=TRAIN_VIDEO_DIR,
        output_dir="mmai-train",
        skip_missing_answers=False,  # train should have all answers
    )

    print(f"\nProcessing TEST split...")
    test_records = process_split(
        csv_path=TEST_CSV,
        video_dir=TEST_VIDEO_DIR,
        output_dir="mmai-test",
        skip_missing_answers=True,  # skip questions without answers
    )

    print(f"\n--- Summary ---")
    print(f"Train: {len(train_records)} entries in mmai-train/data.jsonl")
    print(f"Test:  {len(test_records)} entries in mmai-test/data.jsonl")
    print(f"\nDone! Upload mmai-train/ and mmai-test/ to Colab.")
