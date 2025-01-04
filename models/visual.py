import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
class ImageAnalyzer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def load_and_preprocess(self, image_path):
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform(img).unsqueeze(0).to(self.device)

    def normalize_and_save_gradients(self,f1_grad, f2_grad, save_path1='f1_grad.jpg', save_path2='f2_grad.jpg',
                                     gamma=0.7, contrast_factor=1.3):
        """
        Normalize gradient tensors and enhance contrast

        Args:
            f1_grad, f2_grad: Gradient tensors
            save_path1, save_path2: Save paths
            gamma: Gamma correction value (<1 makes image brighter, >1 makes darker)
            contrast_factor: Contrast enhancement factor (>1 increases contrast)
        """
        # Convert to numpy
        f1_grad = f1_grad.detach().cpu().numpy()
        f2_grad = f2_grad.detach().cpu().numpy()

        # Handle dimensions
        if len(f1_grad.shape) == 4:
            f1_grad = f1_grad[0]
        if len(f2_grad.shape) == 4:
            f2_grad = f2_grad[0]

        if len(f1_grad.shape) == 3:
            f1_grad = np.mean(f1_grad, axis=0)
        if len(f2_grad.shape) == 3:
            f2_grad = np.mean(f2_grad, axis=0)

        # Ensure 2D arrays
        assert len(f1_grad.shape) == 2, f"Unexpected shape for f1_grad: {f1_grad.shape}"
        assert len(f2_grad.shape) == 2, f"Unexpected shape for f2_grad: {f2_grad.shape}"

        def enhance_image(img):
            # 1. Normalize to 0-1
            img_min = np.min(img)
            img_max = np.max(img)
            if img_max > img_min:
                img_norm = (img - img_min) / (img_max - img_min)
            else:
                img_norm = np.zeros_like(img)

            # 2. Apply gamma correction
            img_gamma = np.power(img_norm, gamma)

            # 3. Enhance contrast
            mean = np.mean(img_gamma)
            img_contrast = (img_gamma - mean) * contrast_factor + mean

            # 4. Clip to 0-1 range
            img_contrast = np.clip(img_contrast, 0, 1)

            # 5. Convert to uint8
            return (img_contrast * 255).astype(np.uint8)

        # Enhance and save images
        f1_enhanced = enhance_image(f1_grad)
        f2_enhanced = enhance_image(f2_grad)

        # Save images
        success1 = cv2.imwrite(save_path1, f1_enhanced)
        success2 = cv2.imwrite(save_path2, f2_enhanced)

        # Print debug information
        print(f"f1_grad shape: {f1_grad.shape}")
        print(f"f2_grad shape: {f2_grad.shape}")
        print(f"f1_enhanced min/max: {np.min(f1_enhanced)}/{np.max(f1_enhanced)}")
        print(f"f2_enhanced min/max: {np.min(f2_enhanced)}/{np.max(f2_enhanced)}")
        print(f"Save status - f1: {success1}, f2: {success2}")

        return f1_enhanced, f2_enhanced
    def cal_pixel_sf(self, f1, f2, kernel_radius=5):
        device = f1.device
        b, c, h, w = f1.shape
        r_shift_kernel = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]) \
            .to(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
        b_shift_kernel = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]]) \
            .to(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)

        f1_r_shift = F.conv2d(f1, r_shift_kernel, padding=1, groups=c)
        f1_b_shift = F.conv2d(f1, b_shift_kernel, padding=1, groups=c)
        f2_r_shift = F.conv2d(f2, r_shift_kernel, padding=1, groups=c)
        f2_b_shift = F.conv2d(f2, b_shift_kernel, padding=1, groups=c)

        f1_grad = torch.pow((f1_r_shift - f1), 2) + torch.pow((f1_b_shift - f1), 2)
        f2_grad = torch.pow((f2_r_shift - f2), 2) + torch.pow((f2_b_shift - f2), 2)

        kernel_size = kernel_radius * 2 + 1
        add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float().to(device)
        kernel_padding = kernel_size // 2
        f1_sf = torch.sum(F.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c), dim=1, keepdim=True)
        f2_sf = torch.sum(F.conv2d(f2_grad, add_kernel, padding=kernel_padding, groups=c), dim=1, keepdim=True)

        f1_norm, f2_norm = self.normalize_and_save_gradients(f1_sf, f2_sf)

        weight_zeros = torch.zeros(f1_sf.shape).to(device)
        weight_ones = torch.ones(f1_sf.shape).to(device)
        dm_tensor = torch.where(f1_sf > f2_sf, weight_ones, weight_zeros)
        return dm_tensor

    def cal_edge_maps(self, x1, x2):
        device = x1.device

        sobel_x = torch.FloatTensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]]).to(device)

        sobel_y = torch.FloatTensor([[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]]).to(device)

        sobel_x = sobel_x.reshape(1, 1, 3, 3)
        sobel_y = sobel_y.reshape(1, 1, 3, 3)

        def detect_edges(img):
            if img.shape[1] == 3:
                gray = 0.2989 * img[:, 0:1] + 0.5870 * img[:, 1:2] + 0.1140 * img[:, 2:3]
            else:
                gray = img

            grad_x = F.conv2d(gray, sobel_x, padding=1)
            grad_y = F.conv2d(gray, sobel_y, padding=1)
            grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            grad_magnitude = grad_magnitude / grad_magnitude.max()
            return grad_magnitude

        edge_map1 = detect_edges(x1)
        edge_map2 = detect_edges(x2)
        return edge_map1, edge_map2

    def visualize_and_save_results(self, img1_path, img2_path, save_dir):
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # Load and process images
        img1 = self.load_and_preprocess(img1_path)
        img2 = self.load_and_preprocess(img2_path)

        # Calculate dm_tensor and edge maps
        dm_tensor = self.cal_pixel_sf(img1, img2)
        edge_map1, edge_map2 = self.cal_edge_maps(img1, img2)

        # Convert tensors to numpy arrays for visualization
        def tensor_to_numpy(tensor):
            return tensor.squeeze().cpu().numpy()

        # 创建并保存每张图片
        # 1. 原始图像1
        plt.figure(figsize=(5, 5))
        plt.imshow(tensor_to_numpy(img1).transpose(1, 2, 0))
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, '1_original_image1.jpg'),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

        # 2. 原始图像2
        plt.figure(figsize=(5, 5))
        plt.imshow(tensor_to_numpy(img2).transpose(1, 2, 0))
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, '2_original_image2.jpg'),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

        # 3. DM张量
        plt.figure(figsize=(5, 5))
        plt.imshow(tensor_to_numpy(dm_tensor), cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, '3_dm_tensor.jpg'),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

        # 4. 边缘图1
        plt.figure(figsize=(5, 5))
        plt.imshow(tensor_to_numpy(edge_map1), cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, '4_edge_map1.jpg'),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

        # 5. 边缘图2
        plt.figure(figsize=(5, 5))
        plt.imshow(tensor_to_numpy(edge_map2), cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, '5_edge_map2.jpg'),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

        # 6. 反转DM张量
        plt.figure(figsize=(5, 5))
        plt.imshow(1 - tensor_to_numpy(dm_tensor), cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, '6_inverse_dm_tensor.jpg'),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"所有图片已保存到目录: {save_dir}")

# Usage example:
if __name__ == "__main__":
    analyzer = ImageAnalyzer()
    img1_path = r'E:\ImagefusionDatasets\three_datasets_MFF\Lytro\A\5.jpg'
    img2_path = r'E:\ImagefusionDatasets\three_datasets_MFF\Lytro\B\5.jpg'
    save_dir = r'./analysis_results'  # 指定保存目录
    analyzer.visualize_and_save_results(img1_path, img2_path, save_dir)