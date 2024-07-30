import torch
from torch import nn
import torch.nn.functional as F
from .SubBlocks import conv3x3
import torch.nn.init as init

class DnCNN_Residual(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN_Residual, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        print('init weight')

class UNet_Residual(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=4, wf=64, slope=0.2):
        super(UNet_Residual, self).__init__()
        """
        Reference:
        Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical
        Image Segmentation. MICCAI 2015.
        ArXiv Version: https://arxiv.org/abs/1505.04597

        Args:
            in_channels (int): number of input channels, Default 3
            depth (int): depth of the network, Default 4
            wf (int): number of filters in the first layer, Default 32
        """
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*wf, slope))
            prev_channels = (2**i) * wf

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2**i)*wf, slope))
            prev_channels = (2**i)*wf

        self.last = conv3x3(prev_channels, out_channels, bias=True)

    def forward(self, x):
        y = x
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])
        out=self.last(x)
        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        print('init weight')



class NestedUNet_4_Residual(nn.Module):
    def __init__(self, input_channels=3, out_channels=6, **kwargs):
        super(NestedUNet_4_Residual, self).__init__()
        nb_filter = [64, 128, 256, 512]

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0], slope=0.2)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], slope=0.2)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], slope=0.2)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], slope=0.2)
#######################
        self.conv0_1 = VGGBlock(nb_filter[1], nb_filter[0], nb_filter[0], slope=0.2)
        self.up1_0 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2, bias=True)

        self.conv1_1 = VGGBlock(nb_filter[2], nb_filter[1], nb_filter[1], slope=0.2)
        self.up2_0 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2, bias=True)

        self.conv2_1 = VGGBlock(nb_filter[3], nb_filter[2], nb_filter[2], slope=0.2)
        self.up3_0 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2, bias=True)
#######################
        self.conv0_2 = VGGBlock(nb_filter[0]*3, nb_filter[0], nb_filter[0], slope=0.2)
        self.up1_1 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2, bias=True)

        self.conv1_2 = VGGBlock(nb_filter[1]*3, nb_filter[1], nb_filter[1], slope=0.2)
        self.up2_1 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2, bias=True)
######################
        self.conv0_3 = VGGBlock(nb_filter[0]*4, nb_filter[0], nb_filter[0], slope=0.2)
        self.up1_2 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2, bias=True)
#####################
        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)


    def forward(self, input):
        y = input
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(F.avg_pool2d(x0_0,2))

        up1_0 = self.up1_0(x1_0)
        crop0_0 = center_crop(x0_0, up1_0.shape[2:])
        out = torch.cat([up1_0, crop0_0], 1)
        x0_1 = self.conv0_1(out)
###################
        x2_0 = self.conv2_0(F.avg_pool2d(x1_0,2))

        up2_0 = self.up2_0(x2_0)
        crop1_0 = center_crop(x1_0, up2_0.shape[2:])
        out = torch.cat([up2_0, crop1_0], 1)
        x1_1= self.conv1_1(out)

        up1_1 = self.up1_1(x1_1)
        crop0_0 = center_crop(x0_0, up1_1.shape[2:])
        crop0_1 = center_crop(x0_1, up1_1.shape[2:])
        out = torch.cat([up1_1, crop0_0,crop0_1], 1)
        x0_2= self.conv0_2(out)
##################################
        x3_0 = self.conv3_0(F.avg_pool2d(x2_0,2))

        up3_0 = self.up3_0(x3_0)
        crop2_0 = center_crop(x2_0, up3_0.shape[2:])
        out = torch.cat([up3_0, crop2_0], 1)
        x2_1= self.conv2_1(out)

        up2_1 = self.up2_1(x2_1)
        crop1_0 = center_crop(x1_0, up2_1.shape[2:])
        crop1_1 = center_crop(x1_1, up2_1.shape[2:])
        out = torch.cat([up2_1, crop1_0,crop1_1], 1)
        x1_2= self.conv1_2(out)

        up1_2 = self.up1_2(x1_2)
        crop0_0 = center_crop(x0_0, up1_2.shape[2:])
        crop0_1 = center_crop(x0_1, up1_2.shape[2:])
        crop0_2 = center_crop(x0_2, up1_2.shape[2:])
        out = torch.cat([up1_2, crop0_0,crop0_1,crop0_2], 1)
        x0_3= self.conv0_3(out)
        output = self.final(x0_3)
        return y-output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        print('init weight')

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels,slope=0.2):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, bias=True)
        self.relu1=nn.LeakyReLU(slope, inplace=True)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.LeakyReLU(slope, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.relu2(out)

        return out
def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer
def center_crop(layer, target_size):
    _, _, layer_height, layer_width = layer.size()
    diff_y = (layer_height - target_size[0]) // 2
    diff_x = (layer_width - target_size[1]) // 2
    return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

class UNetUpBlock(nn.Module):
    def __init__(self, in_size,middle_size, out_size, slope=0.2):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = VGGBlock(in_size, middle_size,out_size, slope)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, slope=0.2):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True))
        block.append(nn.LeakyReLU(slope, inplace=True))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True))
        block.append(nn.LeakyReLU(slope, inplace=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, slope=0.2):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, slope)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out