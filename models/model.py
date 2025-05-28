import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from thop import profile

class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3),
                         self.avgpool(t4),
                         self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)
        return att1, att2, att3, att4, att5


class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())

    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]


class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5

        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5

        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_

class InceptionDWConv2d(nn.Module):
    def __init__(self, in_channels, out_channels=None, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        self.out_channels = out_channels or in_channels

        # Add 1x1 convolution to adjust channel number if necessary
        if self.out_channels != in_channels:
            self.channel_adj = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        else:
            self.channel_adj = nn.Identity()

        gc = max(1, int(self.out_channels * branch_ratio))  # Ensure gc is at least 1

        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.split_indexes = (max(0, self.out_channels - 3 * gc), gc, gc, gc)

    def forward(self, x):
        x = self.channel_adj(x)
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )
class UltraLight_UNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
                 split_att='fc', bridge=False):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels * 2+4, c_list[0], 3, stride=1, padding=1)
        )
        # self.encoder1 = nn.Sequential(
        #     nn.Conv2d(input_channels * 2 + 4, c_list[0], 3, stride=1, padding=1)
        # )
        self.encoder2 = nn.Sequential(
            InceptionDWConv2d(c_list[0], c_list[1], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )
        self.encoder3 = nn.Sequential(
            InceptionDWConv2d(c_list[1], c_list[2], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )
        self.encoder4 = nn.Sequential(
            InceptionDWConv2d(c_list[2], c_list[3], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )
        self.encoder5 = nn.Sequential(
            InceptionDWConv2d(c_list[3], c_list[4], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )
        self.encoder6 = nn.Sequential(
            InceptionDWConv2d(c_list[4], c_list[5], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )
        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')


        self.decoder1 = nn.Sequential(
            InceptionDWConv2d(c_list[5], c_list[4], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )
        self.decoder2 = nn.Sequential(
            InceptionDWConv2d(c_list[4], c_list[3], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )
        self.decoder3 = nn.Sequential(
            InceptionDWConv2d(c_list[3], c_list[2], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )
        self.decoder4 = nn.Sequential(
            InceptionDWConv2d(c_list[2], c_list[1], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )
        self.decoder5 =nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1)


        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Sequential(
            nn.Conv2d(c_list[0], num_classes, kernel_size=3, stride=1, padding=1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        dmap_ini1=self.cal_pixel_sf(x1,x2)
        dmap_ini2=torch.ones_like(dmap_ini1)-dmap_ini1
        edge_maps1,edge_maps2=self.cal_edge_maps(x1,x2)
        #
        x = torch.cat((x1, x2,dmap_ini1,dmap_ini2,edge_maps1,edge_maps2), dim=1)

        # x = torch.cat((x1, x2),dim=1)#都不加
        # x = torch.cat((x1, x2,edge_maps1,edge_maps2), dim=1)  # 加DM

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out

        if self.bridge:
            t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out))

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        return torch.sigmoid(out0)

    def cal_pixel_sf(self,f1, f2, kernel_radius=5):
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

        weight_zeros = torch.zeros(f1_sf.shape).to(device)
        weight_ones = torch.ones(f1_sf.shape).to(device)
        dm_tensor = torch.where(f1_sf > f2_sf, weight_ones, weight_zeros)
        return dm_tensor

    def cal_edge_maps(self,x1, x2):
        """
        Calculate edge maps for two input tensors using Sobel operators.

        Args:
            f1: First input tensor (B, C, H, W)
            f2: Second input tensor (B, C, H, W)

        Returns:
            Tuple of two tensors containing the edge maps (edge_map1, edge_map2)
        """
        device = x1.device

        # Define Sobel kernels
        sobel_x = torch.FloatTensor([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]]).to(device)

        sobel_y = torch.FloatTensor([[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]]).to(device)

        # Reshape kernels for conv2d
        sobel_x = sobel_x.reshape(1, 1, 3, 3)
        sobel_y = sobel_y.reshape(1, 1, 3, 3)

        def detect_edges(img):
            # Convert to grayscale if image is RGB
            if img.shape[1] == 3:
                # RGB to grayscale conversion
                gray = 0.2989 * img[:, 0:1] + 0.5870 * img[:, 1:2] + 0.1140 * img[:, 2:3]
            else:
                gray = img

            # Apply Sobel filters
            grad_x = F.conv2d(gray, sobel_x, padding=1)
            grad_y = F.conv2d(gray, sobel_y, padding=1)

            # Calculate gradient magnitude
            grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

            # Normalize to [0, 1]
            grad_magnitude = grad_magnitude / grad_magnitude.max()

            return grad_magnitude

        # Calculate edge maps for both images
        edge_map1 = detect_edges(x1)
        edge_map2 = detect_edges(x2)

        return edge_map1, edge_map2

# Example usage
if __name__ == "__main__":
    # Create a sample input tensor
    batch_size = 1
    channels = 3
    height = 256
    width = 256
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)

    # Initialize the model
    model = UltraLight_UNet()

    # Forward pass
    output = model(x1, x2)

    print(f"Input shape: {x1.shape}")
    print(f"Output shape: {output.shape}")

    # Calculate the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params / 1e6:.2f} M")

    # Calculate FLOPs using thop
    flops, params = profile(model, inputs=(x1, x2))
    print(f"FLOPs: {flops / 1e9:.2f} G")
    print(f"Parameters: {params / 1e6:.2f} M")
