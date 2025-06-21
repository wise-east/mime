import os
from tqdm import tqdm 
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to mime dataset directory")
    args = parser.parse_args()

    SOURCE_VIDEOS_DIR = os.path.join(args.dataset_dir, "videos/test")
    CROPPED_VIDEOS_DIR = os.path.join(args.dataset_dir, "resized_videos/test")
    os.makedirs(CROPPED_VIDEOS_DIR, exist_ok=True)

    video_files = os.listdir(SOURCE_VIDEOS_DIR)
    for video_file in tqdm(video_files):
        if not video_file.endswith('mp4'):
            continue
        video_path = os.path.join(SOURCE_VIDEOS_DIR, video_file)
        output_path = os.path.join(CROPPED_VIDEOS_DIR, video_file)
        if os.path.exists(output_path):
            continue
        #import pdb; pdb.set_trace()
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

            # Resize the frame to 640x480
            resized_frame = cv2.resize(frame, (crop_width, crop_height))

            # Write the cropped frame to the output video
            out.write(resized_frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()