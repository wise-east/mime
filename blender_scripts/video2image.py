# Convert a video to three images and stitch them together horizontally 

import numpy as np
import os
import ffmpeg
from PIL import Image
from pathlib import Path
import click

package_dir = Path(__file__).parent.parent

def video2image(video_path:str, output_dir:str, output_fname:str="", n_images:int=3, direction:str = "horizontal"):
    # Read the video file
    video = ffmpeg.input(video_path)
    # if output_dir ends with /, remove it
    if output_dir.endswith('/'):
        output_dir = output_dir[:-1]

    assert Path(video_path).exists(), f"Video file does not exist: {video_path}"

    # select n_images frames from the video, evenly spaced
    # Get video duration and fps using ffprobe
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    duration = float(probe['format']['duration'])
    fps = eval(video_info['r_frame_rate'])  # Convert fraction string to float
    total_frames = int(duration * fps)

    # Calculate frame indices to extract
    frame_indices = np.linspace(0, total_frames-1, n_images+2, dtype=int)
    timestamps = frame_indices / fps
    timestamps = timestamps[1:-1]
    
    temp_output_dir = Path(output_dir) / Path(output_fname).stem 
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames at the selected timestamps
    for i, timestamp in enumerate(timestamps):
        (
            ffmpeg
            .input(video_path, ss=timestamp)
            .output(f"{temp_output_dir}/frame_{i:04d}.png", vframes=1)
            .run()
        )

    # sample images
    images = [Image.open(f"{temp_output_dir}/frame_{i:04d}.png") for i in range(n_images)]

    if direction == "horizontal": 
        # stitch the images together horizontally    
        stitched_image = Image.new("RGB", (images[0].width * n_images, images[0].height))
        for i, image in enumerate(images):
            stitched_image.paste(image, (i * image.width, 0))
    else: 
        # stitch the images together vertically
        stitched_image = Image.new("RGB", (images[0].width, images[0].height * n_images))
        for i, image in enumerate(images):
            stitched_image.paste(image, (0, i * image.height))

    output_fname = output_fname if output_fname else "stitched_image.png"
    stitched_image.save(f"{output_dir}/{output_fname}")
        
    # remove the extracted frames
    for i in range(n_images):
        os.remove(f"{temp_output_dir}/frame_{i:04d}.png")
    os.rmdir(temp_output_dir)

@click.command()
@click.option("--video_path", type=str, required=True, help="Path to the video file or directory")
@click.option("--output_dir", type=str, default=f"{package_dir}/data/image_inputs", help="Path to the output directory")
@click.option("--output_fname", type=str, default="", help="Name of the output file")
@click.option("--n_images", type=int, default=3, help="Number of images to extract")
@click.option("--direction", type=str, default="horizontal", help="Direction to stitch the images")
def main(video_path, output_dir, output_fname, n_images, direction): 
    
    if output_fname == "": 
        output_fname = Path(video_path).stem + f"_{direction}_{n_images}.png" 
    
    # if video_path is a directory, process all videos in the directory 
    if os.path.isdir(video_path):
        for video_file in os.listdir(video_path):
            video2image(f"{video_path}/{video_file}", output_dir, output_fname, n_images, direction)
    else: 
        video2image(video_path, output_dir, output_fname, n_images, direction)
    
    return output_fname

if __name__ == "__main__":
    main()
