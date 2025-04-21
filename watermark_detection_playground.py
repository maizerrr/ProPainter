import cv2
import numpy as np
import tqdm
from shapely.geometry import Polygon

# Load the pre-trained EAST text detection model
net = cv2.dnn.readNet("weights/frozen_east_text_detection.pb")

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
frame_no = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1
    (H, W) = frame.shape[:2]
    newW, newH = 320, 320
    rW = W / float(newW)
    rH = H / float(newH)
    frame_resized = cv2.resize(frame, (newW, newH))

    blob = cv2.dnn.blobFromImage(frame_resized, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)

    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    (rects, confidences) = decode_predictions(scores, geometry)
    indices = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)

    if len(indices) > 0:
        frame_regions[frame_no] = []
        for i in indices.flatten():
            (x, y, w, h) = rects[i]
            x = int(x * rW)
            y = int(y * rH)
            w = int(w * rW)
            h = int(h * rH)
            frame_regions[frame_no].append((x, x + w, y, y + h))

    # Display the result
    for region in frame_regions.get(frame_no, []):
        (x1, x2, y1, y2) = region
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Unify regions across frames
unified_regions = unify_regions(frame_regions)

# Release resources
cap.release()
cv2.destroyAllWindows()