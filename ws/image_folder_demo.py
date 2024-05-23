# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import os
import cv2
from mmengine.model.utils import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model
from mmseg.apis.inference import show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('folder', help='path to the image folder')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--show', action='store_true', help='Whether to show draw result')
    parser.add_argument(
        '--show-wait-time', default=1, type=int, help='Wait time after imshow')
    parser.add_argument(
        '--output-folder', default=None, type=str, help='Output folder')
    # parser.add_argument(
    #     '--output-fourcc',
    #     default='MJPG',
    #     type=str,
    #     help='Fourcc of the output video')
    # parser.add_argument(
    #     '--output-fps', default=-1, type=int, help='FPS of the output video')
    parser.add_argument(
        '--output-height',
        default=-1,
        type=int,
        help='Frame height of the output video')
    parser.add_argument(
        '--output-width',
        default=-1,
        type=int,
        help='Frame width of the output video')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    assert args.show or args.output_folder, \
        'At least one output should be enabled.'

    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    image_files = list()
    for item in os.listdir(args.folder):
        if item.endswith('.jpg') or item.endswith('.png'):
            image_files.append(item)
    image_files.sort()

    # start looping
    for image_file in image_files:
        # test a single image
        print('Processing {} ...'.format(image_file))
        img = os.path.join(args.folder, image_file)
        result = inference_model(model, img)

        # blend raw image and prediction
        draw_img = show_result_pyplot(model, img, result, show=False)

        if args.show:
            cv2.imshow('images_demo', draw_img)
            cv2.waitKey(args.show_wait_time)

        if args.output_folder:
            # save 'draw_img'
            cv2.imwrite(os.path.join(args.output_folder, image_file), draw_img)

        # cv2.waitKey(10)



if __name__ == '__main__':
    main()
