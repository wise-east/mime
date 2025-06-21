

function download_images() {

    # man angle 0 90 180 270 in blank
    scp hjcho@endeavour.usc.edu://project/jonmay_1426/hjcho/mime/mime/data/videos_transparent/man_Justin_Basketball001_angle0/frame_0220.png  ~/Downloads/man_angle0_frame_0220.png
    scp hjcho@endeavour.usc.edu://project/jonmay_1426/hjcho/mime/mime/data/videos_transparent/man_Justin_Basketball001_angle90/frame_0220.png  ~/Downloads/man_angle90_frame_0220.png
    scp hjcho@endeavour.usc.edu://project/jonmay_1426/hjcho/mime/mime/data/videos_transparent/man_Justin_Basketball001_angle180/frame_0220.png  ~/Downloads/man_angle180_frame_0220.png
    scp hjcho@endeavour.usc.edu://project/jonmay_1426/hjcho/mime/mime/data/videos_transparent/man_Justin_Basketball001_angle270/frame_0220.png  ~/Downloads/man_angle270_frame_0220.png

    # man aligned and misaligned backgrounds 
    scp hjcho@endeavour.usc.edu://project/jonmay_1426/hjcho/mime/mime/data/videos_transparent/man_Justin_Basketball001_angle0/temp_frames_living_room/frame_0220.png  ~/Downloads/man_angle0_living_room_frame_0220.png
    scp hjcho@endeavour.usc.edu://project/jonmay_1426/hjcho/mime/mime/data/videos_transparent/man_Justin_Basketball001_angle0/temp_frames_basketball_court/frame_0220.png  ~/Downloads/man_angle0_basketball_court_frame_0220.png

    # woman in blank  
    scp hjcho@endeavour.usc.edu://project/jonmay_1426/hjcho/mime/mime/data/videos_transparent/woman_Justin_Basketball001_angle0/frame_0220.png  ~/Downloads/woman_angle0_basketball_court_frame_0220.png

    # spacesuits in blank, aligned and misaligned backgrounds 
    scp hjcho@endeavour.usc.edu://project/jonmay_1426/hjcho/mime/mime/data/videos_transparent/spacesuit_Justin_Basketball001_angle0/frame_0220.png  ~/Downloads/spacesuit_angle0_frame_0220.png
    scp hjcho@endeavour.usc.edu://project/jonmay_1426/hjcho/mime/mime/data/videos_transparent/spacesuit_Justin_Basketball001_angle0/temp_frames_living_room/frame_0220.png  ~/Downloads/spacesuit_angle0_living_room_frame_0220.png
    scp hjcho@endeavour.usc.edu://project/jonmay_1426/hjcho/mime/mime/data/videos_transparent/spacesuit_Justin_Basketball001_angle0/temp_frames_basketball_court/frame_0220.png  ~/Downloads/spacesuit_angle0_basketball_court_frame_0220.png
}


download_images()

dimension=480
# Crop all images to square dimensions
for img in ~/Downloads/*_frame_0220.png; do
    # set new image name with ending to dimension x dimension and save to Downloads directory
    new_img_name="$(dirname "$img")/$(basename "$img" .png)_${dimension}x${dimension}.png"
    
    # Get original image dimensions
    dimensions=$(identify -format "%wx%h" "$img")
    echo "Processing $img (Original size: $dimensions)"

    # for angle rotations 90 set offset to the left 
    # for angle rotations 270 set offset to the right
    offset=0
    if [[ $img == *"angle90"* ]]; then
        offset=-160
    elif [[ $img == *"angle270"* ]]; then
        offset=160
    fi

    convert "$img" -gravity center -crop ${dimension}x${dimension}+${offset}+0 +repage "$new_img_name"
    
    # Verify new image dimensions
    new_dimensions=$(identify -format "%wx%h" "$new_img_name")
    echo "Created $new_img_name (New size: $new_dimensions)"
done







