import cv2
import os
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────
VIDEO_DIR   = "Celeb-synthesis"   # folder containing your 795 videos
OUTPUT_DIR  = "images/fakeTestImages"          # drops straight into your existing pipeline
TARGET_FRAMES = 5000                    # total frames to extract across all videos

# ─────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1 — collect all videos
video_exts = {".mp4", ".avi", ".mov", ".mkv"}
videos = [
    f for f in Path(VIDEO_DIR).iterdir()
    if f.suffix.lower() in video_exts
]

if not videos:
    print(f" No videos found in {VIDEO_DIR}")
    exit()

print(f"📹 Found {len(videos)} videos")

# Step 2 — figure out how many frames per video
frames_per_video = TARGET_FRAMES // len(videos)
print(f"🎯 Extracting ~{frames_per_video} frames per video to reach {TARGET_FRAMES} total")

# Step 3 — extract frames
total_extracted = 0

for video_path in tqdm(videos, desc="Processing videos"):
    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        continue

    # Evenly space the frame indices across the video
    if total_frames <= frames_per_video:
        # Video is shorter than target — take every frame
        frame_indices = list(range(total_frames))
    else:
        # Pick evenly spaced indices
        step = total_frames // frames_per_video
        frame_indices = list(range(0, total_frames, step))[:frames_per_video]

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Name: videoname_frameindex.jpg
        fname = f"{video_path.stem}_frame{idx:05d}.jpg"
        out_path = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(out_path, frame)
        total_extracted += 1

    cap.release()

print(f"\n Done — extracted {total_extracted:,} frames to {OUTPUT_DIR}")
print(f"   Now run imageProcessing.py to preprocess them")