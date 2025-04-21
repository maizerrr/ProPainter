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

# Helper function to unify text regions across frames
def unify_text_regions(raw_regions):
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

def detect_static_regions(processed_frames, threshold=100, min_bbox_size=(5, 5), neighbor_threshold=20):
    """
    Detects static regions (e.g., watermark) across all frames and wraps them with non-overlapping bounding boxes.

    Args:
        processed_frames (list): List of frames (color).
        threshold (int): Variance threshold to identify static regions.
        min_bbox_size (tuple): Minimum width and height of a bounding box (width, height).
        neighbor_threshold (int): Maximum distance to consider two bounding boxes as neighbors.

    Returns:
        dict: A dictionary where keys are frame numbers and values are lists of non-overlapping bounding boxes
              [(x1, y1, x2, y2)] for detected static regions.
    """
    # Split frames into color channels (R, G, B)
    channels = [np.stack([frame[:, :, c] for frame in processed_frames], axis=0) for c in range(3)]  # Shape: (num_frames, height, width)

    # Calculate pixel-wise variance for each channel
    variance_maps = [np.var(channel_stack, axis=0) for channel_stack in channels]

    # Combine variance maps (e.g., take the maximum variance across channels)
    combined_variance_map = np.max(variance_maps, axis=0)

    # Threshold the combined variance map to find static regions
    static_region = (combined_variance_map < threshold).astype(np.uint8)  # Binary mask of static regions

    # Find contours of the static region
    contours, _ = cv2.findContours(static_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare the output in the format of frame_regions
    static_regions = {}
    bounding_boxes = []

    if contours:
        # Extract bounding boxes for all contours
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, x + w, y + h))  # Store as (x1, y1, x2, y2)

        # Filter small bounding boxes only if they have no neighbors
        filtered_boxes = []
        for box in bounding_boxes:
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1

            if width >= min_bbox_size[0] and height >= min_bbox_size[1]:
                # Large enough, keep it
                filtered_boxes.append(box)
            else:
                # Check for neighbors that meet the minimum size requirement
                has_valid_neighbor = False
                for other_box in bounding_boxes:
                    if box == other_box:
                        continue
                    ox1, oy1, ox2, oy2 = other_box
                    o_width, o_height = ox2 - ox1, oy2 - oy1

                    # Only consider neighbors that meet the minimum size requirement
                    if o_width >= min_bbox_size[0] and o_height >= min_bbox_size[1]:
                        # Check if the boxes are within the neighbor threshold
                        if abs(x1 - ox2) <= neighbor_threshold or abs(x2 - ox1) <= neighbor_threshold:
                            if abs(y1 - oy2) <= neighbor_threshold or abs(y2 - oy1) <= neighbor_threshold:
                                has_valid_neighbor = True
                                break

                if has_valid_neighbor:
                    filtered_boxes.append(box)

        # Merge overlapping bounding boxes
        merged_boxes = merge_bounding_boxes(filtered_boxes)

        # Add the merged bounding boxes for all frames
        for frame_no in range(1, len(processed_frames) + 1):
            static_regions[frame_no] = merged_boxes

    return static_regions


def merge_bounding_boxes(boxes):
    """
    Merges overlapping bounding boxes into non-overlapping ones and removes boxes
    that are completely contained within larger boxes.

    Args:
        boxes (list): List of bounding boxes [(x1, y1, x2, y2)].

    Returns:
        list: List of merged non-overlapping bounding boxes [(x1, y1, x2, y2)].
    """
    if not boxes:
        return []

    # Sort boxes by their top-left corner (x1, y1)
    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))

    merged_boxes = []
    while boxes:
        # Take the first box and compare it with the rest
        current_box = boxes.pop(0)
        x1, y1, x2, y2 = current_box

        # Check for overlaps with the remaining boxes
        i = 0
        while i < len(boxes):
            bx1, by1, bx2, by2 = boxes[i]

            # Check if the current box overlaps with this box
            if bx1 <= x2 and bx2 >= x1 and by1 <= y2 and by2 >= y1:
                # Merge the boxes
                x1 = min(x1, bx1)
                y1 = min(y1, by1)
                x2 = max(x2, bx2)
                y2 = max(y2, by2)

                # Remove the merged box from the list
                boxes.pop(i)
            else:
                # No overlap, move to the next box
                i += 1

        # Add the merged box to the result
        merged_boxes.append((x1, y1, x2, y2))

    # Remove boxes that are completely contained within others
    final_boxes = []
    for box in merged_boxes:
        x1, y1, x2, y2 = box
        is_contained = False
        for other_box in merged_boxes:
            if box == other_box:
                continue
            ox1, oy1, ox2, oy2 = other_box
            # Check if the current box is completely inside another box
            if x1 >= ox1 and y1 >= oy1 and x2 <= ox2 and y2 <= oy2:
                is_contained = True
                break
        if not is_contained:
            final_boxes.append(box)

    return final_boxes

def post_process_regions(static_regions, text_regions, bbox_expand=0):
    """
    Combines static region and text detection bounding boxes, applies optional expansion, 
    and merges overlapping bounding boxes.

    Args:
        static_regions (dict): Bounding boxes from static region detection. 
                               Format: {frame_no: [(x1, y1, x2, y2), ...]}.
        text_regions (dict): Bounding boxes from text detection. 
                             Format: {frame_no: [(x1, y1, x2, y2), ...]}.
        bbox_expand (int): Number of pixels to expand each bounding box on all sides.

    Returns:
        dict: Combined and processed bounding boxes for each frame.
              Format: {frame_no: [(x1, y1, x2, y2), ...]}.
    """
    combined_regions = {}

    for frame_no in range(1, max(len(static_regions), len(text_regions)) + 1):
        # Get bounding boxes from static regions and text regions
        static_boxes = static_regions.get(frame_no, [])
        text_boxes = text_regions.get(frame_no, [])
        
        # Combine both sets of bounding boxes
        all_boxes = static_boxes + text_boxes

        # Expand bounding boxes if bbox_expand > 0
        if bbox_expand > 0:
            expanded_boxes = []
            for box in all_boxes:
                x1, y1, x2, y2 = box
                expanded_boxes.append((
                    max(0, x1 - bbox_expand),  # Ensure x1 doesn't go below 0
                    max(0, y1 - bbox_expand),  # Ensure y1 doesn't go below 0
                    x2 + bbox_expand,
                    y2 + bbox_expand
                ))
            all_boxes = expanded_boxes

        # Merge overlapping bounding boxes
        merged_boxes = merge_bounding_boxes(all_boxes)

        # Store the merged boxes for the current frame
        combined_regions[frame_no] = merged_boxes

    return combined_regions

# Updated video processing loop
text_regions = {}
processed_frames = []
frame_no = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    processed_frames.append(frame)
cap.release()

# Detect static regions (e.g., watermark) across all frames
static_regions = detect_static_regions(processed_frames, threshold=200)

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
        text_regions[frame_no] = []
        for box in boxes:
            # Extract top-left and bottom-right coordinates
            x1, y1 = np.min(box, axis=0)  # Top-left corner
            x2, y2 = np.max(box, axis=0)  # Bottom-right corner
            text_regions[frame_no].append((x1, y1, x2, y2))

# Unify regions across frames
text_regions = unify_text_regions(text_regions)

# Combine and post-process regions
combined_regions = post_process_regions(static_regions, text_regions)

# Replay the video with post-processed results
for frame_no, frame in enumerate(processed_frames, start=1):
    if frame_no in combined_regions:
        for region in combined_regions[frame_no]:
            (x1, y1, x2, y2) = region
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    cv2.imshow("Processed Video", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):  # Adjust delay as needed for playback speed
        break

# Release resources
cv2.destroyAllWindows()