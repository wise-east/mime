#!/bin/bash

#SBATCH --partition=isi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1

# usage: sbatch submit_render_job.sh <start index> <end index> <n parallel jobs> <pattern>
# e.g. sbatch submit_render_job.sh 1 100 8 woman*

module purge
module load apptainer

HOME_DIR="/project/jonmay_1426/hjcho/mime/"

cd $HOME_DIR

n_parallel_commands=$3
pattern=$4

# define function to render a single blend file
function render_blend_file {
    blend_file=$1
    config_name=$(basename $blend_file) # remove the .blend extension
    config_name=${config_name%.blend}
    output_folder="${HOME_DIR}/data/videos_transparent/${config_name}"

    # skip if the output video already exists
    if [ -f ${output_folder}.mp4 ]; then
        echo "Skipping ${config_name} because the output video already exists"
        return
    fi

    # prerequisite: unset TMPDIR; apptainer pull docker://linuxserver/blender
    # Create output directory if it doesn't exist
    mkdir -p ${output_folder}

    # Get the frame range from the blend file
    frame_range=$(apptainer exec --nv ${HOME_DIR}/../blender_version-4.3.0.sif blender -b ${blend_file} -P ${HOME_DIR}/blender_scripts/find_start_end_frames.py)
    start_frame=$(echo $frame_range | cut -d',' -f1)
    end_frame=$(echo $frame_range | cut -d',' -f2 | cut -d' ' -f1)

    echo "starting frame: $start_frame, end frame: $end_frame"

    # Build list of missing frames
    missing_frames=""
    for i in $(seq $start_frame $end_frame); do
        frame_file=$(printf "${output_folder}/frame_%04d.png" $i)
        if [ ! -f "$frame_file" ]; then
            if [ -z "$missing_frames" ]; then
                missing_frames="$i"
            else
                missing_frames="$missing_frames,$i"
            fi
        fi
    done

    if [ ! -z "$missing_frames" ]; then
        echo "Rendering missing frames for ${config_name}: $missing_frames"
        apptainer exec --nv ${HOME_DIR}/../blender_version-4.3.0.sif blender -b ${blend_file} -P ${HOME_DIR}/blender_scripts/set_render_configs.py -o ${output_folder}/frame_#### -F PNG -x 1 -f $missing_frames -- --cycles-device CUDA
    else
        echo "All frames exist for ${config_name}"
    fi
}


# Check if loop mode is enabled (5th argument, default to false)
loop_mode=${5:-false}

# Function to process blend files
process_blend_files() {
    # get *.blend files in data/blend_files/
    blend_files=$(ls ${HOME_DIR}/data/blend_files/${pattern}.blend)

    # print the number of blend files
    echo "There are ${#blend_files[@]} blend files"

    # select blend files from start_index to end_index
    start_index=$1
    end_index=$2
    blend_files=$(echo "$blend_files" | head -n $end_index | tail -n $((end_index - start_index)))

    # Count total number of files for progress bar
    total_files=$(echo "$blend_files" | wc -l)
    current=0

    # loop through several config_names, running N parallel commands at once
    for blend_file in $blend_files
    do
        render_blend_file $blend_file &
        # wait for N parallel commands to finish
        if [ $(jobs -r | wc -l) -ge $n_parallel_commands ]; then
            wait -n
        fi
        
        # Update progress bar
        current=$((current + 1))
        percentage=$((current * 100 / total_files))
        printf "\rProgress: [%-50s] %d%%" $(printf "#%.0s" $(seq 1 $((percentage/2)))) $percentage
    done
    echo "" # New line after progress bar completes
}

if [ "$loop_mode" = "true" ]; then
    while true; do
        process_blend_files $1 $2
        sleep 30
    done
else
    process_blend_files $1 $2
fi
