import os
import cv2
import numpy as np

def generate_mask_from_video(video_path, output_path='mask.jpg'):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame from the video")
        return False

    # Release the video capture object immediately after getting the frame
    cap.release()

    # Let user select ROI (Region of Interest)
    print("Select ROI and press SPACE or ENTER to confirm!")
    print("Press 'c' to cancel selection.")
    roi = cv2.selectROI("Select Area to Mask", frame, showCrosshair=True)
    cv2.destroyAllWindows()

    # Check if selection is valid (width and height should be > 0)
    if roi[2] == 0 or roi[3] == 0:
        print("Error: Invalid selection. Please select a valid rectangle.")
        return False

    # Create black mask with same dimensions as frame
    mask = np.zeros_like(frame)
    
    # Unpack the ROI coordinates (x, y, width, height)
    x, y, w, h = roi
    
    # Set selected region to white
    mask[y:y+h, x:x+w] = (255, 255, 255)

    # Save the mask as JPEG
    cv2.imwrite(output_path, mask)
    print(f"Mask successfully saved to {output_path}")
    return True

if __name__ == "__main__":
    # Define directories
    video_dir = "inputs/object_removal/video"
    mask_dir = "inputs/object_removal/video_mask"

    # Create directories if they don't exist
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Find .mp4 files in the video directory
    mp4_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

    if not mp4_files:
        print(f"No .mp4 files found in {video_dir}")
    else:
        # Use the first .mp4 file found
        video_file = mp4_files[0]
        video_path = os.path.join(video_dir, video_file)

        # Extract the name of the file (without extension) for output
        output_name = os.path.splitext(video_file)[0] + ".jpg"
        output_path = os.path.join(mask_dir, output_name)

        if generate_mask_from_video(video_path, output_path):
            # Optional: Display the generated mask
            mask = cv2.imread(output_path)
            cv2.imshow("Generated Mask", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()