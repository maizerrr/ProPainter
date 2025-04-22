import os
import cv2
import torch
import torchvision
import torch.nn.functional as F
import numpy as np

from model.transnet_v2 import TransNetV2

THRESHOLD = 0.3

def split_with_transnet(frames:torch.Tensor, threshold=0.5):
    model = TransNetV2()
    model.load_state_dict(torch.load("weights/transnetv2-pytorch-weights.pth", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        x = frames.permute(0, 3, 1, 2) # [T,H,W,3] -> [T,3,H,W]
        x = F.interpolate(x, size=(27, 48), mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 3, 1).unsqueeze(0) # [T,3,H,W] -> [B,T,H,W,3]
        y, _ = model(x)
        y = y.squeeze(0) # [B,T,1] -> [T,1]
        y = (y > threshold).to(torch.uint8)
    prev_pred = -1
    start = 0
    for i in range(y.shape[0]):
        pred = y[i].item()
        if (prev_pred == 1 and pred == 0):
            start = i
        if (prev_pred == 0 and pred == 1):
            yield frames[start:i]
        prev_pred = pred
    if prev_pred == 0:
        yield frames[start:]

def split_with_threshold(frames:torch.Tensor):
    x = frames.to(torch.float32) / 127.5 - 1.0
    clips = []
    start_f = 0
    for i in range(1, t):
        diff = torch.mean(torch.abs(x[i] - x[i-1]), dim=(0,1,2)).item()
        if diff > THRESHOLD:
            clips.append(frames[start_f:i])
            start_f = i
    return clips

if __name__ == "__main__":
    video_dir = "inputs/object_removal/video"
    output_dir = "results/tmp"
    mp4_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    if not mp4_files:
        print(f"No .mp4 files found in {video_dir}")
        exit(1)
    video_file = mp4_files[0]
    # video_path = os.path.join(video_dir, video_file)
    video_path = "/Users/temptrip/Movies/sample_video_2_480p.mp4"
    frames, _, info = torchvision.io.read_video(filename=video_path, pts_unit="sec")
    t, h, w, c = frames.shape
    print(frames.shape)
    print(frames.dtype)
    clips = split_with_transnet(frames)

    for i, clip in enumerate(clips):
        clip = clip.numpy()
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