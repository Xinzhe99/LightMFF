# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import re


def Training_DataReader(root_path):
    source_imgA = glob.glob(root_path + '/train/sourceA/*.jpg')
    source_imgB = glob.glob(root_path + '/train/sourceB/*.jpg')
    gt_img = glob.glob(root_path + '/train/decisionmap/*.png')
    return source_imgA, source_imgB, gt_img

def Test_DataReader(root_path):
    source_imgA = glob.glob(root_path + '/test/sourceA/*.jpg')
    source_imgB = glob.glob(root_path + '/test/sourceB/*.jpg')
    gt_img = glob.glob(root_path + '/test/decisionmap/*.png')
    return source_imgA, source_imgB, gt_img

def Predict_DataReader(root_path):
    source_imgA = glob.glob(root_path + '/A/*.jpg')
    source_imgB = glob.glob(root_path + '/B/*.jpg')
    source_imgA.sort(
        key=lambda x: int(str(re.findall("\d+", x.split('/')[-1])[-1])))  # Sort by the number in the file
    source_imgB.sort(
        key=lambda x: int(str(re.findall("\d+", x.split('/')[-1])[-1])))  # Sort by the number in the file name
    return source_imgA, source_imgB

def Sup_DataLoader(args, sourceA_img, sourceB_img, gt_img):
    source_A_B_ImgLoader = torch.utils.data.DataLoader(
        ImageDataset_pair(args, sourceA_img, sourceB_img, gt_img),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    return source_A_B_ImgLoader


class ImageDataset_pair(Dataset):
    def __init__(self, args, imgA_list, imgB_list, gt_list):
        # 添加随机水平和垂直翻转的数据增强
        transforms_ = [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
            transforms.RandomVerticalFlip(p=0.5),  # 50%概率垂直翻转
            transforms.ToTensor()]

        # 对于mask/gt图像，使用相同的随机翻转变换以保持一致性
        transforms_mask = [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor()]

        self.transform = transforms.Compose(transforms_)
        self.transforms_mask = transforms.Compose(transforms_mask)
        self.imgA_list = imgA_list
        self.imgB_list = imgB_list
        self.gt_list = gt_list

        self.imgA_list = sorted(self.imgA_list)
        self.imgB_list = sorted(self.imgB_list)
        self.gt_list = sorted(self.gt_list)

        # 确保是16的倍数
        max_len_multiple_16 = (len(self.imgA_list) // 16) * 16
        self.imgA_list = self.imgA_list[:max_len_multiple_16]
        self.imgB_list = self.imgB_list[:max_len_multiple_16]
        self.gt_list = self.gt_list[:max_len_multiple_16]

    def __getitem__(self, index):
        imgA = Image.open(self.imgA_list[index]).convert('RGB')
        imgB = Image.open(self.imgB_list[index]).convert('RGB')
        img_gt = Image.open(self.gt_list[index]).convert('L')

        # 注意：对于成对的图像，需要使用相同的随机种子来保证相同的翻转
        seed = torch.randint(0, 2 ** 32, (1,))[0].item()

        torch.manual_seed(seed)
        imgA = self.transform(imgA)

        torch.manual_seed(seed)
        imgB = self.transform(imgB)

        torch.manual_seed(seed)
        img_gt = self.transforms_mask(img_gt)

        return (imgA, imgB, img_gt)

    def __len__(self):
        return len(self.imgA_list)


def Predict_DataLoader(args, root_path):
    source_imgA, source_imgB = Predict_DataReader(root_path)
    source_A_B_ImgLoader = torch.utils.data.DataLoader(
        ImageDataset_pair_predict(args, source_imgA, source_imgB),
        batch_size=1, shuffle=False, pin_memory=True)
    return source_A_B_ImgLoader


class ImageDataset_pair_predict(Dataset):
    def __init__(self, args, imgA_list, imgB_list):
        # Define transforms for images
        transforms_ = [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor()
        ]

        self.transform = transforms.Compose(transforms_)

        # Store image lists
        self.imgA_list = imgA_list
        self.imgB_list = imgB_list

    def __getitem__(self, index):
        # Load and transform source images
        imgA = Image.open(self.imgA_list[index]).convert('RGB')  # 这个保持不变,读取彩色图
        imgB = Image.open(self.imgB_list[index]).convert('RGB')  # 改成RGB,也读取彩色图

        # Apply transforms
        imgA = self.transform(imgA)
        imgB = self.transform(imgB)

        return imgA, imgB

    def __len__(self):
        return len(self.imgA_list)