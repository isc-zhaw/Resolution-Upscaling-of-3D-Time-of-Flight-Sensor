import argparse
import numpy as np
import os
import re
from pathlib import Path
import skimage
import cv2

class Middlebury2005Dataset:
    def __init__(self, root='./dataset_raw/', output_folder_name='data/middlebury_2005') -> None:
        self.root = root
        self.output_folder_name = output_folder_name
        self.image_list = []
        self.disparity_list = []
        self.calibration_list = []
        self.sampling_ratios = [1, 2, 4, 8 ,16]

    def create(self):
        self.get_file_list()
        self.create_folder_structure()
        self.read_files()

    def create_folder_structure(self):
        if not os.path.exists(self.output_folder_name):
            os.mkdir(self.output_folder_name)
        for sampling_ratio in self.sampling_ratios:
            path = os.path.join(self.output_folder_name, f'x{sampling_ratio}')
            if not os.path.exists(path):
                os.mkdir(path)

    def get_file_list(self):
        scenes = list(Path(self.root).glob('*'))
        for scene in scenes:
            self.image_list += [str(scene / 'view1.bmp')]
            self.disparity_list += [ str(scene / 'disp1.bmp') ]

    def read_files(self):
        for im_path, disp_path in zip(self.image_list, self.disparity_list):
            name = re.search(r'[/\\]([\w\-]+)[/\\]view\d.bmp', im_path).group(1)
            im = cv2.imread(im_path)
            disp = cv2.imread(disp_path)[..., 0]
            occ = np.where(disp == 0, 0, 1)
            self.downsample_data(im.astype(np.float32), 255 * disp.astype(np.float32), occ.astype(np.float32), name, depth_dtype=np.uint16)
    
    def downsample_data(self, im, depth, occ, name, depth_dtype=np.uint16):
        max_sampling_ratio = np.max(self.sampling_ratios)
        h, w = depth.shape
        h = h - np.mod(h, max_sampling_ratio)
        w = w - np.mod(w, max_sampling_ratio)
        for sampling_ratio in self.sampling_ratios:
            depth_downsampled = skimage.transform.rescale(depth[:h, :w], 1/sampling_ratio, anti_aliasing=True, order=3).astype(depth_dtype)
            rgb_downsampled = skimage.transform.rescale(im[:h, :w], (1/sampling_ratio, 1/sampling_ratio, 1), anti_aliasing=True, order=3).astype(np.uint8)
            occ_downsampled = skimage.transform.rescale(255 * occ[:h, :w].astype(np.float32), 1/sampling_ratio, anti_aliasing=True, order=3).astype(np.uint8)
            cv2.imwrite(os.path.join(self.output_folder_name, f'x{sampling_ratio}', f'depth_{name}.png'), depth_downsampled)
            cv2.imwrite(os.path.join(self.output_folder_name, f'x{sampling_ratio}', f'rgb_{name}.png'), rgb_downsampled)
            cv2.imwrite(os.path.join(self.output_folder_name, f'x{sampling_ratio}', f'mask_{name}.png'), occ_downsampled)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=False, default='data/dataset_2005_raw', type=str, help='path to Middlebury 2005 folder')
    args = parser.parse_args()
    middlebury_dataset = Middlebury2005Dataset(args.path)
    middlebury_dataset.create()