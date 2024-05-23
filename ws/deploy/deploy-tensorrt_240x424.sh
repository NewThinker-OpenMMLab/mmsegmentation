#!/usr/bin/env bash

my_dir=$(cd $(dirname $0) && pwd)
ws_dir=$(dirname $my_dir)
mmseg_repo=$(dirname $ws_dir)

if [ "$mmdeploy_dir" == "" ]; then
  mmdeploy_dir=/dl/mmdeploy
fi
if ! [ -d $mmdeploy_dir ]; then
  echo "The enviorment variable 'mmdeploy_dir' is set to '$mmdeploy_dir', which is not available!"
  echo "You should run 'export mmdeploy_dir=/path/to/you/mmdeploy' before running this script!"
  exit 1
fi


export PYTHONPATH=$mmseg_repo:$mmdeploy_dir:$PYTHONPATH

if [ "$2" == "" ]; then
  echo "Usage: $0  <config_name>  <quant, fp16 or int8>  [output_model_name]  [torch_model_path]"
  echo "Examples: "
  echo "$0 pspnet_r50-d8_4xb2-40k_cityscapes-512x1024  fp16"
  echo "$0 pspnet_r50-d8_4xb4-20k_voc12aug-512x512  fp16"
  echo "$0 pspnet_r50-d8_4xb4-80k_ade20k-512x512  fp16"
  echo "$0 segformer_mit-b5_8xb2-160k_ade20k-512x512  fp16"
  exit 1
fi


set -e


CONFIG_NAME=$1
config_file=`find $mmseg_repo/configs -name "$CONFIG_NAME.py"`
echo config_file: $config_file
if [ "$config_file" == "" ]; then
  echo "Can't find config '$CONFIG_NAME'!!"
  exit 1
fi


QUANT=$2
# QUANT=fp16
# QUANT=int8
mmdeploy_config=segmentation_tensorrt-${QUANT}_static-240x424.py
mmdeploy_config_path=$mmdeploy_dir/configs/mmseg/$mmdeploy_config
if ! [ -e $mmdeploy_config_path ]; then
  ln -s $my_dir/extra_deploy_configs/$mmdeploy_config $mmdeploy_config_path
fi

OUTPUT_MODEL_NAME=$3
if [ "$OUTPUT_MODEL_NAME" == "" ]; then
    OUTPUT_MODEL_NAME=$CONFIG_NAME-$QUANT
fi
WORKDIR=$my_dir/output_models/tensorrt_240x424/$OUTPUT_MODEL_NAME
EXAMPLE_IMG=$my_dir/example_images/example_image_240x424.jpg

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


cd $mmseg_repo  # we need the 'data' directory under $mmseg_repo 
python3 $mmdeploy_dir/tools/deploy.py \
    $mmdeploy_config_path \
    $config_file \
    $MODEL_PATH \
    $EXAMPLE_IMG \
    --work-dir ${WORKDIR} \
    --device cuda \
    --dump-info
