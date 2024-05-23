#!/usr/bin/env bash

my_dir=$(cd $(dirname $0) && pwd)
ws_dir=$my_dir
mmseg_repo=$(dirname $ws_dir)

cd $mmseg_repo
pip install -v -e .
