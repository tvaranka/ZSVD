#!/bin/bash

input_folder="videos"
output_folder="compressed_videos"

mkdir -p "$output_folder"

for f in "$input_folder"/*.mp4; do
    filename=$(basename "$f")
    ffmpeg -i "$f" -vcodec libx264 -crf 28 -preset slow "$output_folder/$filename"
done

