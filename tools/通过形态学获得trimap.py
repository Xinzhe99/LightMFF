import cv2
import numpy as np


def generate_trimap(mask, uncertain_width=5):
    """
    生成三值trimap，过渡区域为灰色(128)

    参数:
        mask: 二值mask图像
        uncertain_width: 不确定区域的宽度
    """
    # 确保mask是单通道的
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # 确保mask是二值图像
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 使用默认的矩形结构元素进行膨胀和腐蚀
    dilated = cv2.dilate(mask, None, iterations=uncertain_width)
    eroded = cv2.erode(mask, None, iterations=uncertain_width)

    # 创建trimap
    trimap = mask.copy()

    # 将膨胀和腐蚀之间的区域设为128
    uncertain_region = (dilated != eroded)
    trimap[uncertain_region] = 128

    return trimap


# 读取mask
mask = cv2.imread(r'C:\Users\dell\Desktop\Working\LightMFF\result\binary_mask\5.png')
trimap = generate_trimap(mask, uncertain_width=3)

# # 显示结果
# cv2.imshow('Original Mask', mask)
# cv2.imshow('Trimap', trimap)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 如果需要保存结果
cv2.imwrite('trimap.png', trimap)