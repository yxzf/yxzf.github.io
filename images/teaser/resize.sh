#! /bin/bash

echo "convert file:$1"
convert "$1"  -background none -gravity center -resize $2 -extent $2 "$1"
