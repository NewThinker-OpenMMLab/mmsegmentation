
import argparse
import os
import cv2
import time
import numpy as np
from mmdeploy_runtime import Segmentor


def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use sdk python api')
    parser.add_argument('device_name', help='name of device, cuda or cpu')
    parser.add_argument(
        'model_path',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('image_folder', help='path of an image folder')
    parser.add_argument(
        '--show', action='store_true', help='Whether to show draw result')
    parser.add_argument(
        '--show-wait-time', default=1, type=int, help='Wait time after imshow')
    parser.add_argument(
        '--output-folder', default=None, type=str, help='Output folder')

    args = parser.parse_args()
    return args


def get_palette(num_classes=256):
    state = np.random.get_state()
    # random color
    np.random.seed(42)
    palette = np.random.randint(0, 256, size=(num_classes, 3))
    np.random.set_state(state)
    return [tuple(c) for c in palette]


def main():
    args = parse_args()

    # image_folder = cv2.imread(args.image_folder)

    image_files = list()
    for item in os.listdir(args.image_folder):
        if item.endswith('.jpg') or item.endswith('.png'):
            image_files.append(item)
    image_files.sort()

    segmentor = Segmentor(
        model_path=args.model_path, device_name=args.device_name, device_id=0)

    # start looping

    downsample_img = False
    # downsample_img = True

    if args.output_folder:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)

    for idx, image_file in enumerate(image_files):
        img_path = os.path.join(args.image_folder, image_file)
        origin_img = cv2.imread(img_path)

        if downsample_img:
            # img = cv2.resize(origin_img, (424, 240))
            img = cv2.resize(origin_img, (282, 160))
            # img = cv2.resize(origin_img, (212, 120))
        else:
            img = origin_img

        inference_start_time = time.time()
        seg = segmentor(img)
        if seg.dtype == np.float32:
            seg = np.argmax(seg, axis=0)
        inference_end_time = time.time()
        inference_elapsed_time = inference_end_time - inference_start_time
        print('Processed image {} \t ({}) ... (elapsed {} seconds)'.format(idx, image_file, inference_elapsed_time))

        # continue

        palette = get_palette()
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        if downsample_img:
            resized_color_seg = cv2.resize(color_seg, (848, 480), interpolation = cv2.INTER_AREA)
            img = origin_img * 0.5 + resized_color_seg * 0.5

            # img = img * 0.5 + color_seg * 0.5
        else:
            img = img * 0.5 + color_seg * 0.5

        img = img.astype(np.uint8)
        if args.show:
            cv2.imshow('segmentation', img)
            cv2.waitKey(args.show_wait_time)
        if args.output_folder:
            # save 'draw_img'
            cv2.imwrite(os.path.join(args.output_folder, image_file), img)

        #cv2.imwrite('output_segmentation.png', img)


if __name__ == '__main__':
    main()

