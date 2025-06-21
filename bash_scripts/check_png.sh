# check that all png files are not corrupted 
# usage: bash check_png.sh <pattern>
# e.g. bash check_png.sh "woman*"
module load imagemagick
module load parallel

HOME_DIR="/project/jonmay_1426/hjcho/mime"
pattern=$1

# get all the png files in data/videos_transparent/
pngs=$(ls --color=never ${HOME_DIR}/data/videos_transparent/${pattern}/*.png)

# show progress bar 
total_files=$(echo "$pngs" | wc -l)
current=0

# Function to process a single PNG file
process_png() {
    local png=$1
    if ! identify -quiet -regard-warnings $png > /dev/null 2>&1; then
        echo -e "\n${png} is corrupted"
        # delete the png
        rm -f $png
    fi
}

export -f process_png

# Use parallel to process files, updating progress every 1%
echo "$pngs" | parallel --progress --bar --eta process_png {}

echo -e "\nDone checking ${total_files} files"
