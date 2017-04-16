#! /bin/bash

#echo "convert file:$1"
#convert "$1"  -background none -gravity center -resize 400x250 -extent 400x250 "$1"
files=$(ls $1)
echo $files
for file in $files; do
    convert "$1/$file"  -background none -gravity center -resize $2 -extent $2 "$1/$file"
done
