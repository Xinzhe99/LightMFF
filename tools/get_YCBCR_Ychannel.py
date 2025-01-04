# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import os
import cv2

# 读取文件夹A中的所有jpg图片
img_path = r'C:\Users\dell\Desktop\Working\LightMFF\ablation\add_edge\fusion'
folder_out=r'E:\matlabproject\fusion_eva_new\Objective-evaluation-for-image-fusion-main\result_Y\LightMFF_ablation_3'
if not os.path.exists(folder_out):
    os.makedirs(folder_out)
img_names = os.listdir(img_path)
for img_name in img_names:
    if img_name.endswith('.jpg') or img_name.endswith('.png') or img_name.endswith('.bmp'):
        # 读取图片
        img = cv2.imread(os.path.join(img_path, img_name))

        # 转换成YCbCr格式
        img_YCbCr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        # 提取Y通道
        y, cb, cr = cv2.split(img_YCbCr)

        # 保存Y通道图片到folderB
        cv2.imwrite(os.path.join(folder_out, img_name), y)