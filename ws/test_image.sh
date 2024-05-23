#!/usr/bin/env bash

my_dir=$(cd $(dirname $0) && pwd)
ws_dir=$my_dir
mmseg_repo=$(dirname $ws_dir)
export PYTHONPATH=$mmseg_repo:$PYTHONPATH

if [ "$1" == "" ]; then
  echo "Usage: $0  <config_name>  [intput_image]  [output_image]  [/path/to/model.pth]"
  echo "Examples: "
  echo "$0 pspnet_r50-d8_4xb2-40k_cityscapes-512x1024"
  echo "$0 pspnet_r50-d8_4xb4-20k_voc12aug-512x512"
  echo "$0 pspnet_r50-d8_4xb4-80k_ade20k-512x512"
  echo "$0 segformer_mit-b5_8xb2-160k_ade20k-512x512"
  exit 1
fi

CONFIG_NAME=$1
INPUT_IMAEG=$2
OUTPUT_IMAGE=$3
MODEL_PATH=$4

if [ "$MODEL_PATH" == "" ]; then
  if ! [ -d $ws_dir/models/$CONFIG_NAME ]; then
    echo "You haven't downloaded the model file for the config '$CONFIG_NAME'!!"
    exit 1
  fi
  MODEL_PATH=`find $ws_dir/models/$CONFIG_NAME -name *.pth`
fi

echo "MODEL_PATH: $MODEL_PATH"
if [ "$MODEL_PATH" == "" ]; then
  echo "Can't find the model file for the config '$CONFIG_NAME'!!"
  exit 1
fi

config_file=`find $mmseg_repo/configs -name "$CONFIG_NAME.py"`
echo config_file: $config_file
if [ "$config_file" == "" ]; then
  echo "Can't find config '$CONFIG_NAME'!!"
  exit 1
fi

if [ "$INPUT_IMAEG" == "" ]; then
  INPUT_IMAEG=$mmseg_repo/demo/demo.png
fi

if [ "$OUTPUT_IMAGE" == "" ]; then
  OUTPUT_IMAGE=$my_dir/result.jpg
fi

python3 $mmseg_repo/demo/image_demo.py \
  $INPUT_IMAEG $config_file $MODEL_PATH \
  --device cuda:0 --out-file $OUTPUT_IMAGE
