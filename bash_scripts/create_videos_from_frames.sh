#!/bin/bash

#SBATCH --partition=isi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1

module purge
module load apptainer
module load gcc/13.3.0 ffmpeg/7.0
module load conda 
source ~/.bashrc
conda deactivate
conda deactivate
conda activate mime
which pip

# usage: ./create_videos_from_frames.sh <background_config> <pattern> <reverse>

# check for correct number of arguments
if [ $# -lt 2 ]; then
    echo "Usage: ./create_videos_from_frames.sh <background_config> <pattern> <reverse> (e.g., ./create_videos_from_frames.sh blank man*)"
    exit 1
fi

HOME_DIR=/project/jonmay_1426/hjcho/mime
cd ${HOME_DIR}

# list all directories in data/videos_transparent 
pattern=$2
reverse=${3:-false}
frame_dir_pattern=${HOME_DIR}/data/videos_transparent/${pattern}/

if [ ${reverse} = true ]; then
    frame_dirs=$(ls --color=never -d ${frame_dir_pattern} | tac)
else
    frame_dirs=$(ls --color=never -d ${frame_dir_pattern})
fi

# check that the directory has all frames based on the blend file with the same name 
base_path_to_blend_files=${HOME_DIR}/data/blend_files

# check that the background file exists 
if [ ! -f ${background_path} ]; then
    echo "Error: ${background_path} does not exist"
    exit 1
fi

total_count=$(ls -d ${frame_dir_pattern} | wc -l)
completion_count=0
skip_count=0

process_video() {
    local frame_dir=$1
    local background_name=$2
    local base_path_to_blend_files=$3
    local HOME_DIR=$4

    # remove last backslash
    frame_dir=$(echo ${frame_dir} | sed 's/\/$//')

    # create video from frames
    video_name=$(basename ${frame_dir})
    video_path=${HOME_DIR}/data/videos_transparent/${video_name}_${background_name}.mp4

    # if video already exists, skip 
    if [ -f ${video_path} ]; then
        echo "Skipping ${video_path} because it already exists"
        # upload to s3 if it doesn't exist in s3 
        if ! aws s3 ls "s3://mime-understanding/${video_path}" > /dev/null 2>&1; then
            aws s3 cp ${video_path} s3://mime-understanding
            return 0
        fi 
    fi

    # last part of the frame_dir is the name of the blend file 
    blend_file_name=$(basename ${frame_dir})
    blend_file=$(ls --color=never ${base_path_to_blend_files}/${blend_file_name}.blend)
    num_frames=$(ls ${frame_dir}/*.png | wc -l)

    frame_range=$(apptainer exec --nv ${HOME_DIR}/blender_latest.sif blender -b ${blend_file} -P ${HOME_DIR}/blender_scripts/find_start_end_frames.py)
    start_frame=$(echo $frame_range | cut -d',' -f1)
    end_frame=$(echo $frame_range | cut -d',' -f2 | cut -d' ' -f1)

    if [ $num_frames -ne $((end_frame - start_frame + 1)) ]; then
        echo $frame_dir
        echo $num_frames
        echo $start_frame
        echo $end_frame

        echo "Error: ${frame_dir} has ${num_frames} frames, but ${blend_file} has ${end_frame - start_frame + 1} frames"
        return 1
    fi

    command="python ${HOME_DIR}/blender_scripts/overlay_video_on_background.py \
        --frames ${frame_dir} \
        --background ${HOME_DIR}/data/background/${background_name}.png \
        --output ${video_path}"
    echo "Running command: ${command}"
    ${command}
    
    # upload to s3 
    aws s3 cp ${video_path} s3://mime-understanding

    return 0
}

# Process videos in parallel with a maximum of N concurrent jobs
max_parallel=16
current_jobs=0

# set background configuration  
background_config=$1 # one of blank, aligned, misaligned

# load filtered_mocap_info.csv
filtered_mocap_info_path=${HOME_DIR}/data/filtered_mocap_info.csv
filtered_mocap_info=$(cat ${filtered_mocap_info_path})
# create dictionary that maps blend file to background name 
# first column of filtered_mocap_info is blend file name, seventh column is aligned background name, eighth column is misaligned background name
declare -A background_dict

while IFS=, read -r blend_file _ _ _ _ _ aligned_background_name misaligned_background_name _ _ _; do
    # If your CSV contains a header, you may want to skip it by checking for a specific value in blend_file.
    if [ "$blend_file" = "blender_name" ]; then
        continue
    fi

    # action name is what comes after / in the blend file and remove .fbx 
    action_name=$(echo "$blend_file" | cut -d'/' -f2 | sed 's/.fbx//')

    if [ "$background_config" = "aligned" ]; then
        background_dict["$action_name"]="$aligned_background_name"
    else
        background_dict["$action_name"]="$misaligned_background_name"
    fi
done < "$filtered_mocap_info_path"

for key in "${!background_dict[@]}"; do
    echo "$key => ${background_dict[$key]}"
done

for frame_dir in $frame_dirs; do
    # Wait if we've reached max parallel jobs
    if [ $current_jobs -ge $max_parallel ]; then
        wait -n
        current_jobs=$((current_jobs - 1))
    fi

    if [ ${background_config} = "blank" ]; then
        background_name="blank"
    else
        # find the action name in the frame_dir, which is the third item when split by _ 
        action_name=$(basename ${frame_dir} | cut -d'_' -f3)
        background_name=${background_dict[${action_name}]}
    fi

    echo "Processing ${frame_dir} with background ${background_name}"
    process_video "$frame_dir" "$background_name" "$base_path_to_blend_files" "$HOME_DIR" &
    current_jobs=$((current_jobs + 1))
    
    completion_count=$((completion_count + 1))
    echo "Started processing ${completion_count}/${total_count} videos for ${background_name} using frames with pattern ${frame_dir_pattern}"
done

# Wait for remaining jobs to complete
wait

echo "Completed processing all videos"