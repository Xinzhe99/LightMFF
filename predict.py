# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

import torch
import argparse
from Dataloader import Predict_DataLoader
from tools.config_dir import config_dir
import os.path
import cv2
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from models.get_model import get_model
import Dataloader
import skimage.morphology
from skimage import morphology


class ImageProcessor:
    def __init__(self, ks=5, area_ratio=0.01):
        self.ks = ks
        self.area_ratio = area_ratio

    def process_decision_map(self, dm):
        # Create structural element
        se = skimage.morphology.disk(self.ks)
        h, w = dm.shape[:2]

        # Apply morphological operations
        dm = skimage.morphology.binary_opening(dm, se)
        dm = morphology.remove_small_holes(dm == 0, self.area_ratio * h * w)
        dm = np.where(dm, 0, 1)
        dm = skimage.morphology.binary_closing(dm, se)
        dm = morphology.remove_small_holes(dm == 1, self.area_ratio * h * w)
        dm = np.where(dm, 1, 0)

        dm_opt = np.clip(dm, 0, 1)
        return dm_opt


def optimize_mask(mask, use_morphology=True):
    if not use_morphology:
        # Original simple thresholding
        mask_opt = np.zeros_like(mask)
        mask_opt[mask >= 0.5] = 1
        mask_opt[mask < 0.5] = 0
        return mask_opt
    else:
        # Apply morphological processing
        processor = ImageProcessor()
        # First apply threshold
        binary_mask = (mask >= 0.5).astype(float)
        # Then apply morphological operations
        return processor.process_decision_map(binary_mask)


def predict(args, image_path, stict_path, mask_save_path, binary_mask_save_path,
            opt_mask_save_path, fusion_save_path):
    # Create output directories
    os.makedirs(mask_save_path, exist_ok=True)
    os.makedirs(fusion_save_path, exist_ok=True)
    os.makedirs(binary_mask_save_path, exist_ok=True)
    os.makedirs(opt_mask_save_path, exist_ok=True)

    # Load model
    model = get_model(args)
    model.load_state_dict(torch.load(stict_path))
    model.eval()

    # Get data loader and image lists
    dataloder = Predict_DataLoader(args, image_path)
    sourceA_img_list, sourceB_img_list = Dataloader.Predict_DataReader(image_path)

    for i, (imgA, imgB) in tqdm(enumerate(dataloder)):
        # Move all inputs to GPU
        imgA = Variable(imgA).cuda()
        imgB = Variable(imgB).cuda()

        # Forward pass with all four inputs
        mask = model(imgA, imgB)
        mask = mask.detach().cpu().numpy()[0, 0, :, :]  # numpy[0-1]

        # Read source images
        img1 = cv2.imread(sourceA_img_list[i]).astype(np.float32) / 255.0
        img2 = cv2.imread(sourceB_img_list[i]).astype(np.float32) / 255.0
        target_size = (img2.shape[1], img2.shape[0])  # (width, height)

        # Resize mask to match image size
        mask = cv2.resize(mask, target_size)

        # Save original continuous mask
        mask_display = (mask * 255).astype(np.uint8)
        mask_display = np.repeat(mask_display[:, :, np.newaxis], 3, axis=2)
        cv2.imwrite(os.path.join(mask_save_path, '{}.png'.format(i + 1)), mask_display)

        # Generate and save binary mask (simple thresholding)
        binary_mask = optimize_mask(mask, use_morphology=False)
        binary_mask_display = (binary_mask * 255).astype(np.uint8)
        binary_mask_display = np.repeat(binary_mask_display[:, :, np.newaxis], 3, axis=2)
        cv2.imwrite(os.path.join(binary_mask_save_path, '{}.png'.format(i + 1)), binary_mask_display)

        # Generate and save optimized mask (with morphological operations)
        mask_opt = optimize_mask(mask, use_morphology=True)
        mask_opt_display = (mask_opt * 255).astype(np.uint8)
        mask_opt_display = np.repeat(mask_opt_display[:, :, np.newaxis], 3, axis=2)
        cv2.imwrite(os.path.join(opt_mask_save_path, '{}.png'.format(i + 1)), mask_opt_display)

        # Generate fusion result using optimized mask
        mask_opt_3d = np.repeat(mask_opt[:, :, np.newaxis], 3, axis=2)
        fusion_result = img1 * mask_opt_3d + img2 * (1 - mask_opt_3d)
        fusion_result = (fusion_result * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(fusion_save_path, '{}.jpg'.format(i + 1)), fusion_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('--input_size', type=tuple, default=256, help='input size for the network')
    parser.add_argument('--model_path', default=r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_pair_fusion/LightMFF/train_runs/train_runs61/Epoch_180/models/model.pth', help='path to the trained model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for prediction')
    parser.add_argument('--GPU_parallelism', type=bool, default=True, help='use multiple GPUs')
    parser.add_argument('--test_dataset_path', default=r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/ROad-MF/Road-MF/Low_resolution', help='path to test dataset')
    args = parser.parse_args()

    predict_save_path = config_dir(resume=False, subdir_name='predict_run')
    mask_path = os.path.join(predict_save_path, 'mask')
    fusion_result_path = os.path.join(predict_save_path, 'fusion')
    binary_mask_path = os.path.join(predict_save_path, 'binary_mask')
    opt_mask_save_path = os.path.join(predict_save_path, 'opt_mask')

    predict(args,
            image_path=args.test_dataset_path,
            stict_path=args.model_path,
            mask_save_path=mask_path,
            binary_mask_save_path=binary_mask_path,
            opt_mask_save_path=opt_mask_save_path,
            fusion_save_path=fusion_result_path)