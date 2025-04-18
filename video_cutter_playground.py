import os
import cv2
import torch
import torchvision
import numpy as np

THRESHOLD = 0.3

if __name__ == "__main__":
    video_dir = "inputs/object_removal/video"
    output_dir = "results/tmp"
    mp4_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    if not mp4_files:
        print(f"No .mp4 files found in {video_dir}")
        exit(1)
    video_file = mp4_files[0]
    video_path = os.path.join(video_dir, video_file)
    frames, _, info = torchvision.io.read_video(filename=video_path, pts_unit="sec")
    t, h, w, c = frames.shape
    frames = frames.to(torch.float32) / 127.5 - 1.0
    print(frames.shape)
    print(frames.dtype)
    clips = []
    start_f = 0
    for i in range(1, t):
        diff = torch.mean(torch.abs(frames[i] - frames[i-1]), dim=(0,1,2)).item()
        if diff > THRESHOLD:
            clips.append(frames[start_f:i])
            start_f = i
    clips.append(frames[start_f:t])

    for i, clip in enumerate(clips):
        clip = ((clip + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
        output_path = os.path.join(output_dir, f"clip_{i:03d}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter(output_path, fourcc, info['video_fps'], (w, h))
        
        # Write each frame
        for frame in clip:
            # Convert RGB to BGR (OpenCV uses BGR format)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Saved clip {i} to {output_path}")