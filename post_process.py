import os
import numpy as np
import cv2
import skimage.morphology
from skimage import morphology

import os
import numpy as np
import cv2
import skimage.morphology
from skimage import morphology

class ImageProcessor:
    def __init__(self, ks, area_ratio):
        self.ks = ks
        self.area_ratio = area_ratio

    def fuse_images(self, img1, img2, dm):
        # Ensure dm is in the range [0, 1]
        dm = dm.astype(float) / 255.0

        # Expand dimensions if necessary
        if dm.ndim == 2 and img1.ndim == 3:
            dm = np.expand_dims(dm, axis=2)

        # Perform weighted fusion
        fused = img1 * dm + img2 * (1 - dm)
        fused = np.clip(fused, 0, 255).astype(np.uint8)

        return fused

    def process_decision_map(self, dm):
        # Convert initial decision map to uint8
        dm_ini = (dm * 255).astype(np.uint8)

        # Get image dimensions
        h, w = dm.shape[:2]

        # Create structural element
        se = skimage.morphology.disk(self.ks)

        # Apply morphological operations
        dm = skimage.morphology.binary_opening(dm, se)
        dm = morphology.remove_small_holes(dm == 0, self.area_ratio * h * w)
        dm = np.where(dm, 0, 1)
        dm = skimage.morphology.binary_closing(dm, se)
        dm = morphology.remove_small_holes(dm == 1, self.area_ratio * h * w)
        dm = np.where(dm, 1, 0)

        dm_opt = np.clip(dm, 0, 1)
        dm_opt = (dm_opt * 255).astype(np.uint8)

        return dm_ini, dm_opt


def process_decision_maps(input_folder, output_folder, original_images_folder_A, original_images_folder_B, ks=5, area_ratio=0.01):
    processor = ImageProcessor(ks, area_ratio)

    # Create output folders if they don't exist
    os.makedirs(os.path.join(output_folder, 'dm_ini'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'dm_opt'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'fused'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'fused_y'), exist_ok=True)
    # Get list of decision map files
    dm_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    for dm_filename in dm_files:
        # Construct paths for decision map and corresponding original images
        dm_path = os.path.join(input_folder, dm_filename)
        original_image_path_A = os.path.join(original_images_folder_A, dm_filename.replace('_opt_mask', '')).replace('png', 'jpg')
        original_image_path_B = os.path.join(original_images_folder_B, dm_filename.replace('_opt_mask', '')).replace('png', 'jpg')

        # Read decision map and original images
        dm = cv2.imread(dm_path, cv2.IMREAD_GRAYSCALE) / 255.0
        original_image_A = cv2.imread(original_image_path_A)
        original_image_B = cv2.imread(original_image_path_B)

        # Process the decision map
        dm_ini, dm_opt = processor.process_decision_map(dm)

        # Fuse images using the optimized decision map
        fused_image = processor.fuse_images(original_image_A, original_image_B, dm_opt)

        # Save the results
        cv2.imwrite(os.path.join(output_folder, 'dm_ini', f"{dm_filename}"), dm_ini)
        cv2.imwrite(os.path.join(output_folder, 'dm_opt', f"{dm_filename}"), dm_opt)
        cv2.imwrite(os.path.join(output_folder, 'fused', f"{dm_filename}"), fused_image)
        # 将图片转换为YCbCr格式
        img_ycbcr = cv2.cvtColor(fused_image, cv2.COLOR_BGR2YCrCb)
        # 仅保留Y通道
        img_y = img_ycbcr[:, :, 0]
        cv2.imwrite(os.path.join(output_folder, 'fused_y', f"{dm_filename}"), img_y)
        print(f"Processed and fused: {dm_filename}")

    print("Processing and fusion complete.")


# Usage
if __name__ == "__main__":
    Epoch=175
    input_folder = r"C:\Users\dell\Desktop\Working\LightMFF\Epoch_{}\eval\Lytro\opt_mask".format(str(Epoch))
    output_folder = r"C:\Users\dell\Desktop\Working\LightMFF\Epoch_{}\eval\Lytro\processed_results".format(str(Epoch))
    original_images_folder_A = r"E:\ImagefusionDatasets\three_datasets_MFF\Lytro\A".format(str(Epoch))
    original_images_folder_B = r"E:\ImagefusionDatasets\three_datasets_MFF\Lytro\B".format(str(Epoch))
    process_decision_maps(input_folder, output_folder, original_images_folder_A, original_images_folder_B)

    input_folder = r"C:\Users\dell\Desktop\Working\LightMFF\Epoch_{}\eval\MFFW\opt_mask".format(str(Epoch))
    output_folder = r"C:\Users\dell\Desktop\Working\LightMFF\Epoch_{}\eval\MFFW\processed_results".format(str(Epoch))
    original_images_folder_A = r"E:\ImagefusionDatasets\three_datasets_MFF\MFFW\A".format(str(Epoch))
    original_images_folder_B = r"E:\ImagefusionDatasets\three_datasets_MFF\MFFW\B".format(str(Epoch))
    process_decision_maps(input_folder, output_folder, original_images_folder_A, original_images_folder_B)

    input_folder = r"C:\Users\dell\Desktop\Working\LightMFF\Epoch_{}\eval\MFI-WHU\opt_mask".format(str(Epoch))
    output_folder = r"C:\Users\dell\Desktop\Working\LightMFF\Epoch_{}\eval\MFI-WHU\processed_results".format(str(Epoch))
    original_images_folder_A = r"E:\ImagefusionDatasets\three_datasets_MFF\MFI-WHU\A".format(str(Epoch))
    original_images_folder_B = r"E:\ImagefusionDatasets\three_datasets_MFF\MFI-WHU\B".format(str(Epoch))
    process_decision_maps(input_folder, output_folder, original_images_folder_A, original_images_folder_B)