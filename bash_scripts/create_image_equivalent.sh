module purge
module load conda
module load gcc/13.3.0 ffmpeg/7.0
module load parallel
source ~/.bashrc
conda activate mime

HOME_DIR="/project/jonmay_1426/hjcho/mime"

pattern=$1
force=${2:-false}
# find all videos in data/videos_transparent that matches the given pattern ($1)
videos=$(ls ${HOME_DIR}/data/videos_transparent/${pattern}/*.mp4)

# Create a function to process each video
process_video() {
    local video=$1
    local output_dir=${HOME_DIR}/data/inputs_image
    local output_fname="$(basename $video .mp4).png"
    local force=$2
    full_path_to_output_file=${output_dir}/${output_fname}

    if [ -f ${full_path_to_output_file} ] && [ ${force} == "false" ]; then
        echo "Skipping $video because it already exists"
        return 0
    fi

    echo "Processing $video"
    
    # create the image equivalent of the video
    python ${HOME_DIR}/blender_scripts/video2image.py --video_path $video --output_dir $output_dir --output_fname $output_fname --n_images 3 --direction horizontal

    aws s3 cp ${full_path_to_output_file} s3://mime-understanding/
}

export -f process_video
export HOME_DIR

# Process videos in parallel using GNU parallel
echo "$videos" | parallel process_video {} ${force}
