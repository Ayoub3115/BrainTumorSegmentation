import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(0.25))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512, dropout=True)

        self.pool = nn.MaxPool2d(2)

        self.middle = DoubleConv(512, 1024, dropout=True)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512, dropout=True)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256, dropout=True)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        mid = self.middle(self.pool(d4))
        u1 = self.up1(mid)
        u1 = torch.cat([u1, d4], dim=1)
        u1 = self.conv1(u1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)
        u2 = self.conv2(u2)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)
        u3 = self.conv3(u3)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1)
        u4 = self.conv4(u4)
        return torch.sigmoid(self.out(u4))
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(0.25))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=64):
        super(UNetPlusPlus, self).__init__()
        nb_filter = [base_filters, base_filters*2, base_filters*4, base_filters*8, base_filters*16]

        self.pool = nn.MaxPool2d(2, 2)

        # Convs
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4], dropout=True)

        self.up1_0 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], 2, stride=2)
        self.conv0_1 = ConvBlock(nb_filter[0]*2, nb_filter[0])

        self.up2_0 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], 2, stride=2)
        self.conv1_1 = ConvBlock(nb_filter[1]*2, nb_filter[1])

        self.up3_0 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], 2, stride=2)
        self.conv2_1 = ConvBlock(nb_filter[2]*2, nb_filter[2])

        self.up4_0 = nn.ConvTranspose2d(nb_filter[4], nb_filter[3], 2, stride=2)
        self.conv3_1 = ConvBlock(nb_filter[3]*2, nb_filter[3])

        # Nested convs
        self.up1_1 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], 2, stride=2)
        self.conv0_2 = ConvBlock(nb_filter[0]*3, nb_filter[0])

        self.up2_1 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], 2, stride=2)
        self.conv1_2 = ConvBlock(nb_filter[1]*3, nb_filter[1])

        self.up3_1 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], 2, stride=2)
        self.conv2_2 = ConvBlock(nb_filter[2]*3, nb_filter[2])

        self.up1_2 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], 2, stride=2)
        self.conv0_3 = ConvBlock(nb_filter[0]*4, nb_filter[0])

        self.up2_2 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], 2, stride=2)
        self.conv1_3 = ConvBlock(nb_filter[1]*4, nb_filter[1])

        self.up1_3 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], 2, stride=2)
        self.conv0_4 = ConvBlock(nb_filter[0]*5, nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], dim=1))

        return torch.sigmoid(self.final(x0_4))