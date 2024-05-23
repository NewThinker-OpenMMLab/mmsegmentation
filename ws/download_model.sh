#!/usr/bin/env bash

my_dir=$(cd $(dirname $0) && pwd)
ws_dir=$my_dir
mmseg_repo=$(dirname $ws_dir)
mmws_dir=$(dirname $(dirname $mmseg_repo))
# echo "mmws_dir: $mmws_dir"

export PYTHONPATH=$mmseg_repo:$PYTHONPATH

if [ "$1" == "" ]; then
  echo "Usage: $0 <config_name>"
  echo "Examples: "
  echo "$0 pspnet_r50-d8_4xb2-40k_cityscapes-512x1024"
  echo "$0 pspnet_r50-d8_4xb4-20k_voc12aug-512x512"
  echo "$0 pspnet_r50-d8_4xb4-80k_ade20k-512x512"
  echo "$0 segformer_mit-b5_8xb2-160k_ade20k-512x512"
  exit 1
fi

CONFIG_NAME=$1
download_dir=$ws_dir/models/$CONFIG_NAME
mkdir -p $download_dir
cd $mmseg_repo
mim download mmsegmentation --config $CONFIG_NAME --dest $download_dir
