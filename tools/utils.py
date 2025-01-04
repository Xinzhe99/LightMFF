# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import torchvision
import torch
import os
from torchvision.utils import save_image, make_grid

def to_image(tensor, i, tag, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    file = path + '/{}.png'.format(str(i) + '_' + tag)

    # 直接归一化到 [0,1] 范围
    tensor = tensor.detach()
    # tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = torch.clamp(tensor, 0, 1)
    # 创建网格
    grid = make_grid(tensor, nrow=4)

    # 直接保存，因为已经在 [0,1] 范围内
    torchvision.utils.save_image(grid, file)


def to_image_mask(tensor, i, tag, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    file = path + '/{}.png'.format(str(i) + '_' + tag)

    # 只需要detach，因为sigmoid已经把值限制在(0,1)了
    tensor = tensor.detach()
    tensor = torch.clamp(tensor, 0, 1)
    # Create the grid
    grid = make_grid(tensor, nrow=4)

    # 直接保存，因为值已经在正确范围内
    torchvision.utils.save_image(grid, file)


def load_best_eval_log(path):
    file_path=os.path.join(path, 'best_eval_log.txt')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            # 从文件中读取日志内容
            # 这里假设日志内容是以键值对的形式存储的，每行一个键值对，以':'分隔键和值
            log_content = f.readlines()
            log = {}
            for line in log_content:
                key, value = line.strip().split(':')
                log[key.strip()] = float(value.strip())  # 假设值是浮点数类型
        return log
    else:
        # 如果文件不存在，返回默认的日志内容
        return {'bestmae_epoch': 0, 'best_mae': 10, 'fm': 0, 'bestfm_epoch': 0, 'best_fm': 0, 'mae': 0}

def save_parameters(args,save_path,resume_mode):
    if resume_mode:
        # 构建参数文件路径
        parameters_file_path = os.path.join(save_path, f'parameters_resume_{args.resume_times}.txt')
    else:
        parameters_file_path = os.path.join(save_path, 'parameters.txt')

    # 写入参数到文件
    with open(parameters_file_path, 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')

def write_loss_logs(epoch, losses, file_path):
    """
    将损失值写入到文本文件
    :param epoch: 当前 epoch
    :param total_loss_D1: loss_D1 的值
    :param total_loss_D2: loss_D2 的值
    :param total_loss_mask: loss_mask 的值
    :param total_loss_GAN1: loss_GAN1 的值
    :param total_loss_GAN2: loss_GAN2 的值
    :param file_path: 文本文件路径
    """
    # 打开文本文件以写入数据
    # 打开文本文件以写入数据
    with open(file_path, 'a') as f:
        # 写入标量数据
        f.write(f'Epoch: {epoch}, ')
        for loss_name, loss_value in losses.items():
            f.write(f'{loss_name}: {loss_value:.5f}, ')
        f.write('\n')


def write_eval_logs(epoch, mae, fmeasure, recall, precesion, file_path):
    """
    将标量值写入到文本文件
    :param epoch: 当前 epoch
    :param mae: mae 的值
    :param fmeasure: fmeasure 的值
    :param recall: recall 的值
    :param precesion: precesion 的值
    :param file_path: 文本文件路径
    """
    # 打开文本文件以写入数据
    with open(file_path, 'a') as f:
        # 写入标量数据
        f.write(f'Epoch: {epoch}, ')
        f.write(f'mae: {mae}, ')
        f.write(f'fmeasure: {fmeasure}, ')
        f.write(f'recall: {recall}, ')
        f.write(f'precesion: {precesion}\n')
