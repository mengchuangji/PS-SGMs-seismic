import torch
from torch import nn
import torch.nn.functional as F
# mcj 五层的U_net 注意deep_supervision
# __all__ = ['UNet', 'NestedUNet']
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


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3,out_channels=6, slope=0.2,**kwargs):
        super().__init__()

        nb_filter = [64, 128, 256, 512,1024]

        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0],slope=0.2)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1],slope=0.2)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2],slope=0.2)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3],slope=0.2)
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4],slope=0.2)

        # self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2],slope=0.2)
        # self.conv1_2 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1],slope=0.2)
        # self.conv0_3 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0],slope=0.2)

        self.conv3_1 = VGGBlock(nb_filter[4], nb_filter[3], nb_filter[3], slope)
        self.up4_0 = nn.ConvTranspose2d(nb_filter[4], nb_filter[3], kernel_size=2, stride=2, bias=True)

        self.conv2_2 = VGGBlock(nb_filter[3], nb_filter[2], nb_filter[2],slope)
        self.up3_1 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2, bias=True)

        self.conv1_3 = VGGBlock(nb_filter[2], nb_filter[1], nb_filter[1],slope)
        self.up2_2 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2, bias=True)

        self.conv0_4 = VGGBlock(nb_filter[1], nb_filter[0], nb_filter[0], slope)
        self.up1_3 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2, bias=True)

        self.final = conv3x3(nb_filter[0], out_channels, bias=True)

        # self.conv2_1 = VGGBlock(nb_filter[3]+nb_filter[2], nb_filter[2], nb_filter[2], slope)
        # self.up3_0 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2, bias=True)
        #
        # self.conv1_2 = VGGBlock(nb_filter[2]+nb_filter[1], nb_filter[1], nb_filter[1], slope)
        # self.up2_1 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2, bias=True)
        #
        # self.conv0_3 = VGGBlock(nb_filter[1]+nb_filter[0], nb_filter[0], nb_filter[0], slope)
        # self.up1_2 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2, bias=True)
        #
        # self.final = conv3x3(nb_filter[0], out_channels, bias=True)



    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(F.avg_pool2d(x0_0,2))
        x2_0 = self.conv2_0(F.avg_pool2d(x1_0,2))
        x3_0 = self.conv3_0(F.avg_pool2d(x2_0,2))
        x4_0 = self.conv4_0(F.avg_pool2d(x3_0,2))


        # x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        # x1_2 = self.conv1_2(torch.cat([x1_0, self.up(x2_1)], 1))
        # x0_3 = self.conv0_3(torch.cat([x0_0, self.up(x1_2)], 1))

        up4_0 = self.up4_0(x4_0)
        crop3_0 = center_crop(x3_0, up4_0.shape[2:])
        out = torch.cat([up4_0, crop3_0], 1)
        x3_1= self.conv3_1(out)

        up3_1 = self.up3_1(x3_1)
        crop2_0 = center_crop(x2_0, up3_1.shape[2:])
        out = torch.cat([up3_1, crop2_0], 1)
        x2_2= self.conv2_2(out)

        up2_2 = self.up2_2(x2_2)
        crop1_0 = center_crop(x1_0, up2_2.shape[2:])
        out = torch.cat([up2_2, crop1_0], 1)
        x1_3= self.conv1_3(out)

        up1_3 = self.up1_3(x1_3)
        crop0_0 = center_crop(x0_0, up1_3.shape[2:])
        out = torch.cat([up1_3, crop0_0], 1)
        x0_4= self.conv0_4(out)

        output = self.final(x0_4)
        return output

#mcj deep_supervision=True
class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, out_channels=6, deep_supervision=True,slope=0.2, **kwargs):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]
        #mcj
        # self.deep_supervision = deep_supervision
        # self.deep_supervision = False

        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0], slope=0.2)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], slope=0.2)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], slope=0.2)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], slope=0.2)
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4], slope=0.2)
#######################
        self.conv0_1 = VGGBlock(nb_filter[1], nb_filter[0], nb_filter[0], slope=0.2)
        self.up1_0 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2, bias=True)

        self.conv1_1 = VGGBlock(nb_filter[2], nb_filter[1], nb_filter[1], slope=0.2)
        self.up2_0 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2, bias=True)

        self.conv2_1 = VGGBlock(nb_filter[3], nb_filter[2], nb_filter[2], slope=0.2)
        self.up3_0 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2, bias=True)

        self.conv3_1 = VGGBlock(nb_filter[4], nb_filter[3], nb_filter[3], slope=0.2)
        self.up4_0 = nn.ConvTranspose2d(nb_filter[4], nb_filter[3], kernel_size=2, stride=2, bias=True)


#######################
        self.conv0_2 = VGGBlock(nb_filter[0]*3, nb_filter[0], nb_filter[0], slope=0.2)
        self.up1_1 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2, bias=True)

        self.conv1_2 = VGGBlock(nb_filter[1]*3, nb_filter[1], nb_filter[1], slope=0.2)
        self.up2_1 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2, bias=True)

        self.conv2_2 = VGGBlock(nb_filter[2]*3, nb_filter[2], nb_filter[2], slope=0.2)
        self.up3_1 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2, bias=True)


######################
        self.conv0_3 = VGGBlock(nb_filter[0]*4, nb_filter[0], nb_filter[0], slope=0.2)
        self.up1_2 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2, bias=True)

        self.conv1_3 = VGGBlock(nb_filter[1]*4, nb_filter[1], nb_filter[1], slope=0.2)
        self.up2_2 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2, bias=True)
#####################
        self.conv0_4 = VGGBlock(nb_filter[0]*5, nb_filter[0], nb_filter[0], slope=0.2)
        self.up1_3 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2, bias=True)
#####################


        # if self.deep_supervision:#mcj
        #     self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        #     self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        #     self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        #     self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        # else:
        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)


    def forward(self, input):
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

###################################
        x4_0 = self.conv4_0(F.avg_pool2d(x3_0,2))

        up4_0 = self.up4_0(x4_0)
        crop3_0 = center_crop(x3_0, up4_0.shape[2:])
        out = torch.cat([up4_0, crop3_0], 1)
        x3_1= self.conv3_1(out)

        up3_1 = self.up3_1(x3_1)
        crop2_0 = center_crop(x2_0, up3_1.shape[2:])
        crop2_1 = center_crop(x2_1, up3_1.shape[2:])
        out = torch.cat([up3_1, crop2_0,crop2_1], 1)
        x2_2= self.conv2_2(out)

        up2_2 = self.up2_2(x2_2)
        crop1_0 = center_crop(x1_0, up2_2.shape[2:])
        crop1_1 = center_crop(x1_1, up2_2.shape[2:])
        crop1_2 = center_crop(x1_2, up2_2.shape[2:])
        out = torch.cat([up2_2, crop1_0,crop1_1,crop1_2], 1)
        x1_3= self.conv1_3(out)

        up1_3 = self.up1_3(x1_3)
        crop0_0 = center_crop(x0_0, up1_3.shape[2:])
        crop0_1 = center_crop(x0_1, up1_3.shape[2:])
        crop0_2 = center_crop(x0_2, up1_3.shape[2:])
        crop0_3 = center_crop(x0_3, up1_3.shape[2:])
        out = torch.cat([up1_3, crop0_0,crop0_1,crop0_2,crop0_3], 1)
        x0_4= self.conv0_4(out)


        # x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        # x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        # x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # if self.deep_supervision:
        #     output1 = self.final1(x0_1)
        #     output2 = self.final2(x0_2)
        #     output3 = self.final3(x0_3)
        #     output4 = self.final4(x0_4)
        #     return [output1, output2, output3,output4]
        #
        # else:
        output = self.final(x0_4)
        return output


class NestedUNet_4(nn.Module):
    def __init__(self, input_channels=3, out_channels=6, **kwargs):
        super().__init__()

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
        return output