import cv2
import numpy as np
import click
import os
import subprocess
from pathlib import Path
import re

def overlay_frames_on_background(frames_dir, background_path, output_path, fps=60):
    """
    Overlay a sequence of transparent frames onto a background image and create a video.
    
    Args:
        frames_dir (str): Directory containing the frame images
        background_path (str): Path to the background image
        output_path (str): Path for the output video
        fps (int): Frames per second for the output video
    """
    # Read the background image
    background = cv2.imread(background_path)
    if background is None:
        raise ValueError("Could not load background image")
    
    # Create temporary directory for processed frames
    background_name = background_path.split("/")[-1].split(".")[0]
    temp_dir = f"{frames_dir}/temp_frames_{background_name}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Get list of frame files
    frames_path = Path(frames_dir)
    frame_files = list(frames_path.glob("*.png"))
    # change names with frames_###.png to frames_0###.png only if the name is not already frames_0###.png
    # Extract frame numbers and sort
    for f in frame_files:
        frame_count = int(re.search(r'\d+', f.stem).group())
        frame_count = f"{frame_count:04d}"
        new_name = f.parent / f"frame_{frame_count}.png"
        if f != new_name:
            f.rename(new_name)
            
    frame_files = sorted(list(frames_path.glob("*.png")))
    
    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]), cv2.IMREAD_UNCHANGED)
    if first_frame is None:
        raise ValueError("Could not read first frame")
    
    frame_height, frame_width = first_frame.shape[:2]
    
    # Resize background to match frame dimensions
    background = cv2.resize(background, (frame_width, frame_height))
    
    print(f"Processing {len(frame_files)} frames...")
    
    # Process each frame
    for i, frame_path in enumerate(frame_files):
        output_frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")

        if os.path.exists(output_frame_path):
            continue
    
        # Read frame with alpha channel
        frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
            
        # Convert frame to RGBA if it isn't already
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            
        # Split the RGBA frame into color and alpha channels
        b, g, r, a = cv2.split(frame)
        
        # Normalize alpha channel to range 0-1
        alpha = a.astype(float) / 255
        
        # Create a 3-channel alpha mask
        alpha_mask = cv2.merge([alpha, alpha, alpha])
        
        # Blend the frame with the background using alpha mask
        foreground = cv2.merge([b, g, r])
        blended = cv2.multiply(alpha_mask, foreground.astype(float)) + \
                 cv2.multiply(1.0 - alpha_mask, background.astype(float))
        
        # Convert back to uint8
        blended = blended.astype(np.uint8)
        
        # Save processed frame
        cv2.imwrite(output_frame_path, blended)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(frame_files)} frames")
    
    print("Creating video from processed frames...")
    
    # Use FFmpeg to create MP4 video with H.264 codec
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%04d.png'),
        '-c:v', 'libx264',  # Use H.264 codec
        '-preset', 'medium',  # Encoding preset (slower = better compression)
        '-crf', '23',  # Quality (lower = better quality, 23 is default)
        '-pix_fmt', 'yuv420p',  # Pixel format for better compatibility
        '-movflags', '+faststart',  # Enable streaming
        output_path
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print("Video creation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during video creation: {e}")
        raise
    # finally:
    #     # Clean up temporary files
    #     print("Cleaning up temporary files...")
    #     for file in os.listdir(temp_dir):
    #         os.remove(os.path.join(temp_dir, file))
    #     os.rmdir(temp_dir)

@click.command()
@click.option('-f', '--frames', help='Directory containing the frame images', required=True)
@click.option('-b', '--background', help='Path to the background image', required=True)
@click.option('-o', '--output', help='Path for the output video', required=True)
@click.option('--fps', default=60, help='Frames per second for the output video')
def main(frames, background, output, fps):
    try:
        overlay_frames_on_background(frames, background, output, fps)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()