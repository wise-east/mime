### correct the naming of the png files (frames_###.png to frame_####.png)
./correct_naming.sh man*
./correct_naming.sh woman*
./correct_naming.sh spacesuit*

### check for corrupted png files and delete them
# need to run several commands to check all files because of the large number of files
./check_png.sh man*angle0*
./check_png.sh man*angle90*
./check_png.sh man*angle180*
./check_png.sh man*angle270*
./check_png.sh woman*
./check_png.sh spacesuit*

### check for remainig rendering jobs and run them (starts from scratch or fills in only the missing frames as needed) 
# it will check for all .blend files in data/blend_files that matches the given pattern ($4) and runs the rendering jobs that fall in the given range (start and end index ($1 and $2)) with the given number of parallel jobs ($3) 
# refer to blender_scripts/README.md on how to create the blend files
# has the option to run in loop mode, which will keep running until all jobs are finished, add true to the end of the command to run in loop mode
sbatch submit_render_job.sh 1 100 8 man*angle0*
sbatch submit_render_job.sh 1 100 8 man*angle90*
sbatch submit_render_job.sh 1 100 8 man*angle180*
sbatch submit_render_job.sh 1 100 8 man*angle270*
sbatch submit_render_job.sh 1 100 8 woman*
sbatch submit_render_job.sh 1 100 8 spacesuit*

### create videos from frames
# goes through data/videos_transparent and creates videos from the frames that matches the given pattern ($2) with the chosen background ($1) and uploads them to s3
./create_videos_from_frames.sh blank man*angle0* false
./create_videos_from_frames.sh blank man*angle90* false
./create_videos_from_frames.sh blank man*angle180* false
./create_videos_from_frames.sh blank man*angle270* false
./create_videos_from_frames.sh aligned man*angle0* false 
./create_videos_from_frames.sh misaligned man*angle0* false 

./create_videos_from_frames.sh blank spacesuit* false
./create_videos_from_frames.sh aligned spacesuit*angle0* false 
./create_videos_from_frames.sh misaligned spacesuit*angle0* false 

./create_videos_from_frames.sh blank woman* false

### check if the video contains all the frames from the directory with the same name as the video 
# if not, delete the video and ask for confirmation to delete the video from s3
./check_video_correct.sh 

# repeat processes above as necessary until all videos are correct for all .blend files 





