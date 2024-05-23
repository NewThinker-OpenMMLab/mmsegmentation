#!/usr/bin/env bash

my_dir=$(cd $(dirname $0) && pwd)


if [ "$2" == "" ]; then
  echo "Usage: $0  <deployed_model>  <image_folder>  [output_folder]"
  echo "Examples: "
  echo "$0 output_models/tensorrt_240x424/segformer_mit-b5_8xb2-160k_ade20k-512x512-fp16  ../test_images"
  exit 1
fi

DEPLOYED_MODEL=$1
IMAGE_FOLDER=$2
OUTPUT_FOLDER=$3

if [ "$mmdeploy_build_dir" == "" ]; then
  mmdeploy_build_dir=/dl/mmdeploy/build/lib
fi

if ! [ -d $mmdeploy_build_dir ]; then
  echo "The enviorment variable 'mmdeploy_build_dir' is set to '$mmdeploy_build_dir', which is not available!"
  echo "You should run 'export mmdeploy_build_dir=/path/to/you/mmdeploy/build/lib' before running this script!"
  exit 1
fi

if [ "$OUTPUT_FOLDER" == "" ]; then
  OUTPUT_FOLDER=$my_dir/result_images
fi

export PYTHONPATH=$mmdeploy_build_dir:$PYTHONPATH

# segformer_fp16
python3 $my_dir/image_folder_segmentation.py cuda \
    $DEPLOYED_MODEL \
    $IMAGE_FOLDER  \
    --show --output-folder $OUTPUT_FOLDER

