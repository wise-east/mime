# check if the video contains all the frames from the directory with the same name as the video 

# usage: bash check_video_correct.sh 
# e.g. bash check_video_correct.sh
module load ffmpeg

HOME_DIR="/project/jonmay_1426/hjcho/mime"
base_url="https://mime-understanding.s3.amazonaws.com"

# get all the videos in data/videos_transparent/
videos=$(ls --color=never ${HOME_DIR}/data/videos_transparent/*.mp4)

# print the number of videos
echo "Number of videos: $(echo "$videos" | wc -l)"

count=0
for video in $videos; do
    video_name=$(basename $video)
    video_name=${video_name%.mp4}
    # remove the background name from the video name
    dir_name=${video_name%_*}

    # delete video if it is smaller than 10kb 
    if [ $(stat -c "%s" "$video") -lt 10240 ]; then
        echo "${video}: File size is less than 10kb, deleting..."
        rm ${video}
        # ask for confirmation to delete from s3
        read -p "Delete the video from s3? (y/n): ${base_url}/${video_name}.mp4" should_delete
        if [ $should_delete == "y" ]; then
            aws s3 rm s3://mime-understanding/${video_name}.mp4
        fi
        count=$((count + 1))
        continue
    fi

    # if dir doesn't exist, repeat the command one more time (some actions have _ in the name)
    if [ ! -d "${HOME_DIR}/data/videos_transparent/${dir_name}" ]; then
        dir_name=${dir_name%_*}
    fi

    if [ ! -d "${HOME_DIR}/data/videos_transparent/${dir_name}" ]; then
        echo "Directory ${HOME_DIR}/data/videos_transparent/${dir_name} does not exist"
        continue
    fi

    frames_dir=${HOME_DIR}/data/videos_transparent/${dir_name}
    num_frames=$(ls --color=never ${frames_dir}/*.png | wc -l)

    num_frames_in_video=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 ${video})
    if [ $num_frames -ne $num_frames_in_video ]; then
        echo "${video}: Incorrect number of frames: ${num_frames} (in directory) != ${num_frames_in_video} (video)"
        # # delete the video
        rm ${video}
        # ask for confirmation
        read -p "Delete the video from s3? (y/n): ${base_url}/${video_name}.mp4" should_delete
        if [ $should_delete == "y" ]; then
            aws s3 rm s3://mime-understanding/${video_name}.mp4
        fi
        count=$((count + 1))
    else
        echo "${video_name}.mp4: Correct number of frames (${num_frames})"
    fi
done

echo "Number of videos with incorrect number of frames or errors: ${count}"

# check that all videos for all .blend files are present for blank background 
# get all the .blend files in data/blend_files/
blend_files=$(ls --color=never ${HOME_DIR}/data/blend_files/*.blend)

# for blend_file in $blend_files; do
#     blend_file_name=$(basename $blend_file)
#     blend_file_name=${blend_file_name%.blend}