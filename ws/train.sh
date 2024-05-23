#!/usr/bin/env bash

my_dir=$(cd $(dirname $0) && pwd)
ws_dir=$my_dir
mmseg_repo=$(dirname $ws_dir)
export PYTHONPATH=$mmseg_repo:$PYTHONPATH


if [ "$1" == "" ]; then
  echo "Usage: $0  <config_name>"
  echo "Examples: "
  echo "$0 pspnet_r50-d8_4xb2-40k_cityscapes-512x1024"
  echo "$0 pspnet_r50-d8_4xb4-20k_voc12aug-512x512"
  echo "$0 pspnet_r50-d8_4xb4-80k_ade20k-512x512"
  echo "$0 segformer_mit-b5_8xb2-160k_ade20k-512x512"
  exit 1
fi

CONFIG_NAME=$1
config_file=`find $mmseg_repo/configs -name "$CONFIG_NAME.py"`
echo config_file: $config_file
if [ "$config_file" == "" ]; then
  echo "Can't find config '$CONFIG_NAME'!!"
  exit 1
fi


WORK_DIR=$my_dir/training_results/$CONFIG_NAME

#export CUDA_LAUNCH_BLOCKING=1
#export CUDA_LAUNCH_BLOCKING=0

cd $mmseg_repo

######### Train with single GPU

# Choose one CUDA device.
export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=1
# export CUDA_VISIBLE_DEVICES=2

python3 tools/train.py ${config_file} --work-dir ${WORK_DIR}
#python3 tools/train.py ${config_file} --work-dir ${WORK_DIR} \
#    --resume


######### Train with multiple GPUs

# GPU_NUM=3

# bash tools/dist_train.sh \
#     ${config_file}  ${GPU_NUM} \
#     --work-dir  ${WORK_DIR} \
#     --resume
