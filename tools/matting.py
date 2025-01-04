import numpy as np
from scipy.sparse import csr_matrix, linalg
import cv2


class RobustMatting:
    def __init__(self, neighborhood_size=3, num_samples=10):
        self.neighborhood_size = neighborhood_size
        self.num_samples = num_samples

    def get_neighborhood_pixels(self, img, x, y):
        """获取像素点周围的邻域像素"""
        h, w = img.shape[:2]
        r = self.neighborhood_size // 2

        x_min = max(0, x - r)
        x_max = min(w, x + r + 1)
        y_min = max(0, y - r)
        y_max = min(h, y + r + 1)

        return img[y_min:y_max, x_min:x_max].reshape(-1, 3)

    def sample_colors(self, img, trimap):
        """对前景和背景区域进行颜色采样"""
        h, w = img.shape[:2]
        fg_samples = {}  # 前景样本
        bg_samples = {}  # 背景样本

        # 获取前景和背景区域的坐标
        fg_coords = np.where(trimap == 255)
        bg_coords = np.where(trimap == 0)

        # 采样前景颜色
        for y, x in zip(*fg_coords):
            neighborhood = self.get_neighborhood_pixels(img, x, y)
            if len(neighborhood) > 0:
                # 使用K-means进行聚类采样
                if len(neighborhood) > self.num_samples:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    _, labels, centers = cv2.kmeans(neighborhood.astype(np.float32),
                                                    self.num_samples, None, criteria,
                                                    10, cv2.KMEANS_RANDOM_CENTERS)
                    fg_samples[(x, y)] = centers
                else:
                    fg_samples[(x, y)] = neighborhood

        # 采样背景颜色
        for y, x in zip(*bg_coords):
            neighborhood = self.get_neighborhood_pixels(img, x, y)
            if len(neighborhood) > 0:
                if len(neighborhood) > self.num_samples:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    _, labels, centers = cv2.kmeans(neighborhood.astype(np.float32),
                                                    self.num_samples, None, criteria,
                                                    10, cv2.KMEANS_RANDOM_CENTERS)
                    bg_samples[(x, y)] = centers
                else:
                    bg_samples[(x, y)] = neighborhood

        return fg_samples, bg_samples

    def compute_color_distance(self, pixel, samples):
        """计算像素与样本之间的颜色距离"""
        if len(samples) == 0:
            return np.inf

        pixel = pixel.reshape(1, 3)
        distances = np.sqrt(np.sum((samples - pixel) ** 2, axis=1))
        return np.min(distances)

    def get_alpha_from_color_samples(self, pixel, fg_samples, bg_samples):
        """基于颜色样本估计alpha值"""
        fg_dist = self.compute_color_distance(pixel, fg_samples)
        bg_dist = self.compute_color_distance(pixel, bg_samples)

        if fg_dist == np.inf and bg_dist == np.inf:
            return 0.5
        elif fg_dist == np.inf:
            return 0.0
        elif bg_dist == np.inf:
            return 1.0

        alpha = bg_dist / (fg_dist + bg_dist)
        return np.clip(alpha, 0, 1)

    def solve_alpha_matting(self, img, trimap):
        """求解alpha matting"""
        h, w = img.shape[:2]

        # 采样前景和背景颜色
        fg_samples, bg_samples = self.sample_colors(img, trimap)

        # 初始化alpha图
        alpha = np.zeros((h, w), dtype=np.float32)

        # 对未知区域进行alpha估计
        unknown_coords = np.where(trimap == 128)
        for y, x in zip(*unknown_coords):
            pixel = img[y, x]

            # 在局部窗口中寻找最近的前景和背景样本
            local_fg_samples = []
            local_bg_samples = []
            r = self.neighborhood_size * 2

            x_min = max(0, x - r)
            x_max = min(w, x + r + 1)
            y_min = max(0, y - r)
            y_max = min(h, y + r + 1)

            for i in range(x_min, x_max):
                for j in range(y_min, y_max):
                    if (i, j) in fg_samples:
                        local_fg_samples.extend(fg_samples[(i, j)])
                    if (i, j) in bg_samples:
                        local_bg_samples.extend(bg_samples[(i, j)])

            local_fg_samples = np.array(local_fg_samples)
            local_bg_samples = np.array(local_bg_samples)

            # 计算alpha值
            alpha[y, x] = self.get_alpha_from_color_samples(pixel, local_fg_samples, local_bg_samples)

        # 对已知区域直接赋值
        alpha[trimap == 255] = 1
        alpha[trimap == 0] = 0

        return alpha

    def refine_alpha(self, img, alpha, iterations=1):
        """使用双边滤波细化alpha图"""
        refined_alpha = alpha.copy()

        for _ in range(iterations):
            # 使用双边滤波替代导向滤波
            refined_alpha = cv2.bilateralFilter(
                refined_alpha.astype(np.float32),
                d=9,  # 邻域直径
                sigmaColor=75,  # 颜色空间标准差
                sigmaSpace=75  # 坐标空间标准差
            )

        # 确保值在[0,1]范围内
        refined_alpha = np.clip(refined_alpha, 0, 1)
        return refined_alpha

def main():
    # 示例使用
    img = cv2.imread(r'E:\matlabproject\fusion_eva_new\Objective-evaluation-for-image-fusion-main\A\5.jpg')
    trimap = cv2.imread(r'C:\Users\dell\Desktop\Working\LightMFF\code\tools\trimap.png', cv2.IMREAD_GRAYSCALE)

    # 创建Matting对象
    matting = RobustMatting(neighborhood_size=3, num_samples=10)

    # 计算alpha图
    alpha = matting.solve_alpha_matting(img, trimap)

    # 细化alpha图
    refined_alpha = matting.refine_alpha(img, alpha)

    # 保存结果ximgproc
    cv2.imwrite('alpha.png', (refined_alpha * 255).astype(np.uint8))


if __name__ == '__main__':
    main()