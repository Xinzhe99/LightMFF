# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

import os.path
import cv2
import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg
import time
from skimage import feature
from tqdm import tqdm

def make_system(L, sparse_map, constraint_factor=0.001):
    '''

    :param L: Laplacian Matrix.
    Check the paper "A closed-form solution to natural image matting"
    :param sparse_map: Estimated sparse blur values
    :param constraint_factor: Internal parameter for propagation
    :return: System parameters to solve for defocus blur propagation
    '''
    spflatten = sparse_map.ravel()

    D = scipy.sparse.diags(spflatten)

    # combine constraints and graph laplacian
    A = constraint_factor * D + L
    # constrained values of known alpha values
    b = constraint_factor * D * spflatten

    return A, b

def g1x(x, y, s1):
    '''

    :param x:
    :param y:
    :param s1:
    :return:
    '''
    s1sq = s1 ** 2
    g = -1 * np.multiply(np.divide(x, 2 * np.pi * s1sq ** 2),
                         np.exp(-1 * np.divide(x ** 2 + y ** 2, 2 * s1sq)))

    return g


def g1y(x, y, s1):
    '''

    :param x:
    :param y:
    :param s1:
    :return:
    '''
    s1sq = s1 ** 2
    g = -1 * np.multiply(np.divide(y, 2 * np.pi * s1sq ** 2),
                         np.exp(-1 * np.divide(x ** 2 + y ** 2, 2 * s1sq)))

    return g


def get_laplacian(I, r=1):
    '''

    :param I: An RGB image [0-1]
    :param r: radius
    :return: The Laplacian matrix explained in Levin's paper
    '''
    eps = 0.0000001
    h, w, c = I.shape
    wr = (2 * r + 1) * (2 * r + 1)

    M_idx = np.arange(h * w).reshape(w, h).T
    n_vals = (w - 2 * r) * (h - 2 * r) * wr ** 2

    # data for matting laplacian in coordinate form
    row_idx = np.zeros(n_vals, dtype=np.int64)
    col_idx = np.zeros(n_vals, dtype=np.int64)
    vals = np.zeros(n_vals, dtype=np.float64)
    lenr = 0

    for j in range(r, h - r):
        for i in range(r, w - r):
            winr = I[j - r:j + r + 1, i - r:i + r + 1, 2]
            wing = I[j - r:j + r + 1, i - r:i + r + 1, 1]
            winb = I[j - r:j + r + 1, i - r:i + r + 1, 0]
            win_idx = M_idx[j - r:j + r + 1, i - r:i + r + 1].T.ravel()

            meanwinr = winr.mean()
            winrsq = np.multiply(winr, winr)
            varI_rr = winrsq.sum() / wr - meanwinr ** 2

            meanwing = wing.mean()
            wingsq = np.multiply(wing, wing)
            varI_gg = wingsq.sum() / wr - meanwing ** 2

            meanwinb = winb.mean()
            winbsq = np.multiply(winb, winb)
            varI_bb = winbsq.sum() / wr - meanwinb ** 2

            winrgsq = np.multiply(winr, wing)
            varI_rg = winrgsq.sum() / wr - meanwinr * meanwing

            winrbsq = np.multiply(winr, winb)
            varI_rb = winrbsq.sum() / wr - meanwinr * meanwinb

            wingbsq = np.multiply(wing, winb)
            varI_gb = wingbsq.sum() / wr - meanwing * meanwinb

            Sigma = np.array([[varI_rr, varI_rg, varI_rb],
                              [varI_rg, varI_gg, varI_gb],
                              [varI_rb, varI_gb, varI_bb]])

            meanI = np.array([meanwinr, meanwing, meanwinb])

            Sigma = Sigma + eps * np.eye(3)

            winI = np.zeros((wr, c))

            winI[:, 0] = winr.T.ravel()
            winI[:, 1] = wing.T.ravel()
            winI[:, 2] = winb.T.ravel()

            winI = winI - meanI

            inv_cov = np.linalg.inv(Sigma)
            tvals = (1 + np.matmul(np.matmul(winI, inv_cov), winI.T)) / wr

            row_idx[lenr:wr ** 2 + lenr] = np.tile(win_idx, (1, wr)).ravel()
            col_idx[lenr:wr ** 2 + lenr] = np.tile(win_idx, (wr, 1)).T.ravel()
            vals[lenr:wr ** 2 + lenr] = tvals.T.ravel()

            lenr += wr ** 2

    # Lsparse = scipy.sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(h*w, h*w))
    Lsparse = scipy.sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(w * h, w * h))

    row_idx2 = np.zeros(w * h, dtype=np.int64)
    col_idx2 = np.zeros(w * h, dtype=np.int64)
    vals2 = np.zeros(w * h, dtype=np.float64)

    row_idx2[:] = np.arange(w * h)
    col_idx2[:] = np.arange(w * h)
    vals2[:] = Lsparse.sum(axis=1).ravel()

    LDsparse = scipy.sparse.coo_matrix((vals2, (row_idx2, col_idx2)), shape=(w * h, w * h))

    return LDsparse - Lsparse

def convert_float64_to_uint8(edge_map_float64):
    """
    将 float64 类型的边缘图转换为 uint8 类型。

    :param edge_map_float64: 输入的 float64 类型的边缘图，形状为 (H, W)
    :return: 转换后的 uint8 类型的边缘图，形状为 (H, W)
    """
    # 1. 归一化到 [0, 1] 范围
    edge_map_normalized = (edge_map_float64 - np.min(edge_map_float64)) / (np.max(edge_map_float64) - np.min(edge_map_float64))

    # 2. 缩放到 [0, 255] 范围
    edge_map_scaled = edge_map_normalized * 255

    # 3. 类型转换为 uint8
    edge_map_uint8 = edge_map_scaled.astype(np.uint8)

    return edge_map_uint8
def estimate_sparse_blur(gimg, edge_map, std1, std2):
    '''

    :param gimg: Grayscale image
    :param edge_map: An edge map of the image
    :param std1: Standard deviation of reblurring
    :param std2: Standard deviation of second reblurring
    :return: Estimated sparse blur values at edge locations
    '''
    half_window = 11
    m = half_window * 2 + 1
    a = np.arange(-half_window, half_window + 1)
    xmesh = np.tile(a, (m, 1))
    ymesh = xmesh.T

    f11 = g1x(xmesh, ymesh, std1)
    f12 = g1y(xmesh, ymesh, std1)

    f21 = g1x(xmesh, ymesh, std2)
    f22 = g1y(xmesh, ymesh, std2)

    gimx1 = scipy.ndimage.convolve(gimg, f11, mode='nearest')
    gimy1 = scipy.ndimage.convolve(gimg, f12, mode='nearest')
    mg1 = np.sqrt(gimx1 ** 2 + gimy1 ** 2)

    gimx2 = scipy.ndimage.convolve(gimg, f21, mode='nearest')
    gimy2 = scipy.ndimage.convolve(gimg, f22, mode='nearest')
    mg2 = np.sqrt(gimx2 ** 2 + gimy2 ** 2)

    R = np.divide(mg1, mg2)
    R = np.multiply(R, edge_map > 0)

    sparse_vals = np.divide(R ** 2 * (std1 ** 2) - (std2 ** 2), 1 - R ** 2)
    sparse_vals[sparse_vals < 0] = 0

    sparse_bmap = np.sqrt(sparse_vals)
    sparse_bmap[np.isnan(sparse_bmap)] = 0
    sparse_bmap[sparse_bmap > 5] = 5
    return sparse_bmap

def estimate_bmap_laplacian(img, sigma_c, std1, std2):
    '''

    :param img: An RGB image [0-255]
    :param sigma_c: Sigma parameter for Canny edge detector
    :param std1: Standard deviation of reblurring
    :param std2: Standard deviation of second reblurring
    :return: defocus blur map of the given image
    '''
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    edge_map = feature.canny(gimg, sigma_c)
    sparse_bmap = estimate_sparse_blur(gimg, edge_map, std1, std2)
    sparse_bmap_norm =convert_float64_to_uint8(sparse_bmap)

    h, w = sparse_bmap.shape
    L1 = get_laplacian(img / 255.0)
    A, b = make_system(L1, sparse_bmap.T)

    bmap = scipy.sparse.linalg.spsolve(A, b).reshape(w, h).T
    return bmap,sparse_bmap_norm


if __name__ == '__main__':

    t1 = time.time()
    input_folder = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF/Lytro/A'
    output_folder_full = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF/Lytro/A_dbd'
    # output_folder_sparse = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF'

    # 创建输出文件夹
    os.makedirs(output_folder_full, exist_ok=True)
    # os.makedirs(output_folder_sparse, exist_ok=True)

    # 遍历输入文件夹中的所有图片文件
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_folder, filename)
            image_name = os.path.basename(image_path)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (256, 256))
            # 处理图片
            fblurmap, sparse_bmap_norm = estimate_bmap_laplacian(img, sigma_c=1, std1=1, std2=1.5)

            # 保存处理后的图片到对应的文件夹
            cv2.imwrite(os.path.join(output_folder_full, image_name.replace('.jpg', '.png')),
                        np.uint8((fblurmap / fblurmap.max()) * 255))
            # cv2.imwrite(os.path.join(output_folder_sparse, image_name.replace('.jpg', '.png')), sparse_bmap_norm)
    print(f"Total processing time: {time.time() - t1} seconds")

    t1 = time.time()
    input_folder = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF/Lytro/B'
    output_folder_full = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF/Lytro/B_dbd'
    # output_folder_sparse = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF'

    # 创建输出文件夹
    os.makedirs(output_folder_full, exist_ok=True)
    # os.makedirs(output_folder_sparse, exist_ok=True)

    # 遍历输入文件夹中的所有图片文件
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_folder, filename)
            image_name = os.path.basename(image_path)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (256, 256))
            # 处理图片
            fblurmap, sparse_bmap_norm = estimate_bmap_laplacian(img, sigma_c=1, std1=1, std2=1.5)

            # 保存处理后的图片到对应的文件夹
            cv2.imwrite(os.path.join(output_folder_full, image_name.replace('.jpg', '.png')),
                        np.uint8((fblurmap / fblurmap.max()) * 255))
            # cv2.imwrite(os.path.join(output_folder_sparse, image_name.replace('.jpg', '.png')), sparse_bmap_norm)
    print(f"Total processing time: {time.time() - t1} seconds")

    t1 = time.time()
    input_folder = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF/MFFW/A'
    output_folder_full = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF/MFFW/A_dbd'
    # output_folder_sparse = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF'

    # 创建输出文件夹
    os.makedirs(output_folder_full, exist_ok=True)
    # os.makedirs(output_folder_sparse, exist_ok=True)

    # 遍历输入文件夹中的所有图片文件
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_folder, filename)
            image_name = os.path.basename(image_path)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (256, 256))
            # 处理图片
            fblurmap, sparse_bmap_norm = estimate_bmap_laplacian(img, sigma_c=1, std1=1, std2=1.5)

            # 保存处理后的图片到对应的文件夹
            cv2.imwrite(os.path.join(output_folder_full, image_name.replace('.jpg', '.png')),
                        np.uint8((fblurmap / fblurmap.max()) * 255))
            # cv2.imwrite(os.path.join(output_folder_sparse, image_name.replace('.jpg', '.png')), sparse_bmap_norm)
    print(f"Total processing time: {time.time() - t1} seconds")

    t1 = time.time()
    input_folder = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF/MFFW/B'
    output_folder_full = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF/MFFW/B_dbd'
    # output_folder_sparse = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF'

    # 创建输出文件夹
    os.makedirs(output_folder_full, exist_ok=True)
    # os.makedirs(output_folder_sparse, exist_ok=True)

    # 遍历输入文件夹中的所有图片文件
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_folder, filename)
            image_name = os.path.basename(image_path)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (256, 256))
            # 处理图片
            fblurmap, sparse_bmap_norm = estimate_bmap_laplacian(img, sigma_c=1, std1=1, std2=1.5)

            # 保存处理后的图片到对应的文件夹
            cv2.imwrite(os.path.join(output_folder_full, image_name.replace('.jpg', '.png')),
                        np.uint8((fblurmap / fblurmap.max()) * 255))
            # cv2.imwrite(os.path.join(output_folder_sparse, image_name.replace('.jpg', '.png')), sparse_bmap_norm)
    print(f"Total processing time: {time.time() - t1} seconds")


    t1 = time.time()
    input_folder = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF/MFI-WHU/A'
    output_folder_full = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF/MFI-WHU/A_dbd'
    # output_folder_sparse = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF'

    # 创建输出文件夹
    os.makedirs(output_folder_full, exist_ok=True)
    # os.makedirs(output_folder_sparse, exist_ok=True)

    # 遍历输入文件夹中的所有图片文件
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_folder, filename)
            image_name = os.path.basename(image_path)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (256, 256))
            # 处理图片
            fblurmap, sparse_bmap_norm = estimate_bmap_laplacian(img, sigma_c=1, std1=1, std2=1.5)

            # 保存处理后的图片到对应的文件夹
            cv2.imwrite(os.path.join(output_folder_full, image_name.replace('.jpg', '.png')),
                        np.uint8((fblurmap / fblurmap.max()) * 255))
            # cv2.imwrite(os.path.join(output_folder_sparse, image_name.replace('.jpg', '.png')), sparse_bmap_norm)
    print(f"Total processing time: {time.time() - t1} seconds")




    t1 = time.time()
    input_folder = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF/MFI-WHU/B'
    output_folder_full = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF/MFI-WHU/B_dbd'
    # output_folder_sparse = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF'

    # 创建输出文件夹
    os.makedirs(output_folder_full, exist_ok=True)
    # os.makedirs(output_folder_sparse, exist_ok=True)

    # 遍历输入文件夹中的所有图片文件
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_folder, filename)
            image_name = os.path.basename(image_path)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (256, 256))
            # 处理图片
            fblurmap, sparse_bmap_norm = estimate_bmap_laplacian(img, sigma_c=1, std1=1, std2=1.5)

            # 保存处理后的图片到对应的文件夹
            cv2.imwrite(os.path.join(output_folder_full, image_name.replace('.jpg', '.png')),
                        np.uint8((fblurmap / fblurmap.max()) * 255))
            # cv2.imwrite(os.path.join(output_folder_sparse, image_name.replace('.jpg', '.png')), sparse_bmap_norm)
    print(f"Total processing time: {time.time() - t1} seconds")