import cv2
import numpy as np
from tqdm import tqdm
import torch
from shapely.geometry import Polygon
from torch.autograd import Variable

from model.craft import CRAFT, resize_aspect_ratio, normalizeMeanVariance, getDetBoxes, adjustResultCoordinates, copyStateDict
from model.misc import get_device

device = get_device()
print(f"Using device: {device}")

# Load the pre-trained EAST text detection model
net = CRAFT()
net.load_state_dict(copyStateDict(torch.load("weights/craft_mlt_25k.pth", map_location=device)))  # Use device for loading weights
net.to(device)  # Move model to the appropriate device
net.eval()

# Open the video file
cap = cv2.VideoCapture("/Users/temptrip/Movies/sample_video_1_480p.mp4")  # Replace with your video file path

# Helper function to decode predictions
def decode_predictions(scores, geometry, conf_threshold=0.5):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < conf_threshold:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, w, h))
            confidences.append(float(scoresData[x]))

    return (rects, confidences)

# Helper function to unify text regions across frames
def unify_regions(raw_regions):
    keys = sorted(raw_regions.keys())
    unified_regions = {}
    last_key = keys[0]
    unify_value_map = {last_key: raw_regions[last_key]}

    for key in keys[1:]:
        current_regions = raw_regions[key]
        new_unify_values = []

        for idx, region in enumerate(current_regions):
            last_standard_region = unify_value_map[last_key][idx] if idx < len(unify_value_map[last_key]) else None

            if last_standard_region and are_similar(region, last_standard_region):
                new_unify_values.append(last_standard_region)
            else:
                new_unify_values.append(region)

        unify_value_map[key] = new_unify_values
        last_key = key

    for key in keys:
        unified_regions[key] = unify_value_map[key]
    return unified_regions

# Helper function to check similarity between regions
def are_similar(region1, region2):
    xmin1, xmax1, ymin1, ymax1 = region1
    xmin2, xmax2, ymin2, ymax2 = region2

    return abs(xmin1 - xmin2) <= 10 and abs(xmax1 - xmax2) <= 10 and \
           abs(ymin1 - ymin2) <= 10 and abs(ymax1 - ymax2) <= 10

# Updated video processing loop
frame_regions = {}
processed_frames = []
frame_no = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    processed_frames.append(frame)
cap.release()

for frame in tqdm(processed_frames, desc="Processing frames", unit="frame"):
    frame_no += 1
    frame_resized, target_ratio, size_heatmap = resize_aspect_ratio(frame, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(frame_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0).to(device))     # Move tensor to the appropriate device

    with torch.no_grad():
        # Forward pass
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()  # Move back to CPU for further processing
    score_link = y[0,:,:,1].cpu().data.numpy()

    # Post-processing
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold=0.7, link_threshold=0.4, low_text=0.4, poly=False)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    if len(boxes) > 0:
        frame_regions[frame_no] = []
        for box in boxes:
            # Extract top-left and bottom-right coordinates
            x1, y1 = np.min(box, axis=0)  # Top-left corner
            x2, y2 = np.max(box, axis=0)  # Bottom-right corner
            frame_regions[frame_no].append((x1, y1, x2, y2))

# Unify regions across frames
unified_regions = unify_regions(frame_regions)

# Replay the video with post-processed results
for frame_no, frame in enumerate(processed_frames, start=1):
    if frame_no in unified_regions:
        for region in unified_regions[frame_no]:
            (x1, y1, x2, y2) = region
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    cv2.imshow("Processed Video", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):  # Adjust delay as needed for playback speed
        break

# Release resources
cv2.destroyAllWindows()