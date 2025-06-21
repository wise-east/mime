# change all frames_%03d.png to frame_%04d.png 
module load parallel

HOME_DIR="/project/jonmay_1426/hjcho/mime"
pattern=$1

# get all the png files in data/videos_transparent/
pngs=$(ls --color=never ${HOME_DIR}/data/videos_transparent/${pattern}/frames_*.png)

# Function to process a single PNG file
process_png() {
    local png=$1
    # extract frames_
    frame_number=$(echo $png | grep -oP 'frames_\K[0-9]+')
    # convert to decimal number first to handle leading zeros
    frame_number=$((10#$frame_number))
    # format as 4-digit number
    frame_number=$(printf "%04d" $frame_number)
    parent_dir=$(dirname $png)
    new_filename="${parent_dir}/frame_${frame_number}.png"
    # echo "Changing ${png} to ${new_filename}"
    mv $png $new_filename
}

export -f process_png

# Use parallel to process files, showing progress bar
echo "$pngs" | parallel --progress --bar --eta process_png {}
