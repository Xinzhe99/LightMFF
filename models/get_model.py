# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import torch.nn as nn
import torch
from models.model import UltraLight_UNet

def get_model(args):
    model = UltraLight_UNet()
    # generator=Generator(input_channels=1)
    if torch.cuda.is_available():
        model = model.cuda()
    if args.GPU_parallelism:
        model = nn.DataParallel(model)
    return model
