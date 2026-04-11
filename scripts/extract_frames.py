import cv2
import os
from pathlib import Path

REAL_VIDEO_DIR = "datasets/ffpp/real"
FAKE_VIDEO_DIR = "datasets/ffpp/fake"

REAL_FRAME_DIR = "datasets/ffpp/real_frames"
FAKE_FRAME_DIR = "datasets/ffpp/fake_frames"

os.makedirs(REAL_FRAME_DIR, exist_ok=True)
os.makedirs(FAKE_FRAME_DIR, exist_ok=True)


def extract_frames(video_dir, save_dir, every_n=10):
    videos = list(Path(video_dir).glob("*.mp4"))

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        frame_id = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % every_n == 0:
                frame = cv2.resize(frame, (224, 224))

                save_path = os.path.join(
                    save_dir,
                    f"{video_path.stem}_{saved}.jpg"
                )

                cv2.imwrite(save_path, frame)
                saved += 1

            frame_id += 1

        cap.release()


extract_frames(REAL_VIDEO_DIR, REAL_FRAME_DIR)
extract_frames(FAKE_VIDEO_DIR, FAKE_FRAME_DIR)

print("Frame extraction completed")