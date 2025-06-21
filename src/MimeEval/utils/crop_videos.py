import os
from tqdm import tqdm 
import cv2
import argparse

def crop_center(frame, crop_width, crop_height):
    height, width, _ = frame.shape
    start_x = int((width - crop_width) / 2)
    start_y = int((height - crop_height) / 2)
    return frame[start_y:start_y+crop_height, start_x:start_x+crop_width]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to mime dataset directory")
    args = parser.parse_args()

    SOURCE_VIDEOS_DIR = os.path.join(args.dataset_dir, "videos/test")
    CROPPED_VIDEOS_DIR = os.path.join(args.dataset_dir, "cropped_videos/test")
    os.makedirs(CROPPED_VIDEOS_DIR, exist_ok=True)

    video_files = os.listdir(SOURCE_VIDEOS_DIR)
    for video_file in tqdm(video_files):
        if not video_file.endswith('mp4'):
            continue
        video_path = os.path.join(SOURCE_VIDEOS_DIR, video_file)
        output_path = os.path.join(CROPPED_VIDEOS_DIR, video_file)
        if os.path.exists(output_path):
            continue
        crop_width = 640
        crop_height = 480  # Adjust as needed

        #print(video_path)
        cap = cv2.VideoCapture(video_path)

        # Get the video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate crop coordinates
        x_start = (frame_width - crop_width) // 2
        y_start = (frame_height - crop_height) // 2
        x_end = x_start + crop_width
        y_end = y_start + crop_height

        # Define the codec and create the VideoWriter object for MP4
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
        out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))

        # Process the video frame by frame
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Crop the center of the frame
            cropped_frame = frame[y_start:y_end, x_start:x_end]

            # Write the cropped frame to the output video
            out.write(cropped_frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()