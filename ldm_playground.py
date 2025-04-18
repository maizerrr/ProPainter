import os
import PIL
import requests
import torch
import cv2
import numpy as np
from io import BytesIO

from diffusers import StableDiffusionInpaintPipeline

from model.misc import get_device

def extract_image_and_mask(video_path, mask_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame from the video")
        return None, None
    cap.release()

    mask = PIL.Image.open(mask_path)
    mask = np.array(mask.convert("L"))  # Convert to grayscale
    mask[mask>0.1] = 1
    mask[mask<=0.1] = 0

    frame = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mask = PIL.Image.fromarray(mask.astype(np.uint8) * 255)
    return frame, mask

def image_inpainting(image, image_mask, device):
    width, height = image.size
    image = image.resize((512,512))
    image_mask = image_mask.resize((512,512))
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        variant="fp16",
        torch_dtype=torch.float16
    )
    pipe = pipe.to(device)
    if device == "cpu":
        for param in pipe.parameters():
            param.data = param.data.float()
    prompt = "" # dummy prompt
    result = pipe(prompt=prompt, image=image, mask_image=image_mask).images[0]
    return result.resize((width, height))

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    video_dir = "inputs/object_removal/video"
    mask_dir = "inputs/object_removal/video_mask"

    mp4_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

    if not mp4_files:
        print(f"No .mp4 files found in {video_dir}")
        exit(1)
    
    video_file = mp4_files[0]
    video_path = os.path.join(video_dir, video_file)
    mask_file = os.path.splitext(video_file)[0] + ".jpg"
    mask_path = os.path.join(mask_dir, mask_file)

    image, mask = extract_image_and_mask(video_path, mask_path)
    if image is None or mask is None:
        print("Error: Could not extract image and mask.")
        exit(1)

    print(f"Start to process image {video_file} with mask {mask_file}...")
    result = image_inpainting(image, mask, device)
    print("Image inpainting completed.")
    result.show()