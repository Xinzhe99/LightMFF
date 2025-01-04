import os
import cv2

# 源文件夹路径
input_folder = r"C:\Users\dell\Desktop\fusion"
# 目标文件夹路径
output_folder = r"C:\Users\dell\Desktop\24_10_30"

# 确保目标文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历源文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 构建完整的文件路径
    input_path = os.path.join(input_folder, filename)

    # 检查文件是否为图片文件（这里假设图片文件是常见的格式）
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
        # 读取图片
        img = cv2.imread(input_path)

        # 将图片转换为YCbCr格式
        img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        # 仅保留Y通道
        img_y = img_ycbcr[:, :, 0]

        # 构建输出文件路径
        output_path = os.path.join(output_folder, filename)

        # 保存Y通道的图片
        cv2.imwrite(output_path, img_y)

        print(f"Processed and saved: {output_path}")

print("All images processed and saved.")