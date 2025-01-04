import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import re

def natural_sort_key(s):
    """实现自然排序的键函数，提取文件名中的数字进行比较"""
    # 使用正则表达式将字符串分割成数字和非数字部分
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def get_image_pairs(folder_a, folder_b):
    """获取两个文件夹中的图片对，确保它们按数字顺序对应"""
    # 获取支持的图片格式文件
    supported_formats = ('.png', '.jpg', '.jpeg')

    # 使用natural_sort_key函数进行排序
    images_a = sorted(
        [f for f in os.listdir(folder_a) if f.lower().endswith(supported_formats)],
        key=natural_sort_key
    )
    images_b = sorted(
        [f for f in os.listdir(folder_b) if f.lower().endswith(supported_formats)],
        key=natural_sort_key
    )
    # 验证两个文件夹中的图片数量是否匹配
    if len(images_a) != len(images_b):
        raise ValueError(f"文件夹中的图片数量不匹配！A: {len(images_a)}, B: {len(images_b)}")

    # 返回配对的图片列表
    return list(zip(images_a, images_b))
def fusion_images(args):
    """执行图像融合"""
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取图片对
    image_pairs = get_image_pairs(args.folder_a, args.folder_b)

    # 遍历所有图片对进行融合
    for idx, (img_a_name, img_b_name) in enumerate(tqdm(image_pairs, desc="Processing")):
        # 读取图片A和图片B
        img_a = cv2.imread(os.path.join(args.folder_a, img_a_name))
        img_b = cv2.imread(os.path.join(args.folder_b, img_b_name))

        # 读取对应的mask
        mask_name = img_a_name.replace('jpg','png')  # mask的命名格式应该与之前的代码一致
        mask_path = os.path.join(args.mask_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for pair {idx + 1}, skipping...")
            continue

        # 读取mask并确保它是3通道的
        mask = cv2.imread(mask_path)
        if mask is None:
            print(f"Error: Cannot read mask {mask_path}")
            continue

        # 将mask转换为0-1范围的浮点数
        mask = mask.astype(np.float32) / 255.0

        # 确保所有图像尺寸一致
        target_size = (img_b.shape[1], img_b.shape[0])  # (width, height)
        if mask.shape[:2] != img_b.shape[:2]:
            mask = cv2.resize(mask, target_size)
        if img_a.shape[:2] != img_b.shape[:2]:
            img_a = cv2.resize(img_a, target_size)

        # 执行融合
        img_a = img_a.astype(np.float32) / 255.0
        img_b = img_b.astype(np.float32) / 255.0

        # 使用mask进行融合
        fusion_result = img_a * mask + img_b * (1 - mask)

        # 将结果转换回uint8格式
        fusion_result = (fusion_result * 255).astype(np.uint8)

        # 保存融合结果
        output_name = f"{idx + 1}.jpg"
        cv2.imwrite(os.path.join(args.output_dir, output_name), fusion_result)


def main():
    parser = argparse.ArgumentParser(description='Image Fusion Tool')
    parser.add_argument('--folder_a', default=r'E:\matlabproject\fusion_eva_new\Objective-evaluation-for-image-fusion-main\A', help='Path to the first source images folder')
    parser.add_argument('--folder_b', default=r'E:\matlabproject\fusion_eva_new\Objective-evaluation-for-image-fusion-main\B', help='Path to the second source images folder')
    parser.add_argument('--mask_dir', default=r'C:\Users\dell\Desktop\Working\LightMFF\result\binary_mask', help='Path to the mask directory')
    parser.add_argument('--output_dir', default=r'C:\Users\dell\Desktop\Working\LightMFF\result\fusion_ini', help='Path to save fusion results')

    args = parser.parse_args()

    # 检查输入路径是否存在
    if not all(os.path.exists(path) for path in [args.folder_a, args.folder_b, args.mask_dir]):
        raise ValueError("One or more input directories do not exist!")

    fusion_images(args)


if __name__ == '__main__':
    main()