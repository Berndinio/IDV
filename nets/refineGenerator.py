########################################################################################################################
# Jan Sieber, 3219317
# Python 3.5
# Deep Vision, University of Heidelberg, Prof. Dr. Björn Ommer
########################################################################################################################

import torch
import torch.nn as nn
from Utils import Utils


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, linearScaling, last_layer=False):
        super(ResidualBlock, self).__init__()
        self.last_layer = last_layer
        self.relu = nn.ReLU().to(Utils.g_device)

        self.conv1 = nn.Conv2d(in_channel, in_channel + linearScaling, (3, 3), stride=1,
                               padding=1).to(Utils.g_device)
        self.conv2 = nn.Conv2d(in_channel + linearScaling, in_channel + linearScaling, (3, 3), stride=1,
                               padding=1).to(Utils.g_device)
        if not self.last_layer:
            self.conv3 = nn.Conv2d(in_channel + linearScaling, in_channel + linearScaling, (3, 3), stride=2,
                                   padding=1).to(Utils.g_device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        if not self.last_layer:
            x = self.conv3(x)
            x = self.relu(x)
        return x


# mode = 1 or 2
# 1 = With Upsampling (nearest neighbor)
# 2 = With Deconvolution
class ResidualBlockUp(nn.Module):
    def __init__(self, in_channel, linearScaling, last_layer=False, mode=2, outputSize=None):
        super(ResidualBlockUp, self).__init__()
        self.last_layer = last_layer
        self.outputSize = outputSize
        self.mode = mode

        # layers
        self.relu = nn.ReLU().to(Utils.g_device)

        self.conv1 = nn.Conv2d(in_channel, in_channel - linearScaling, (3, 3), stride=1,
                               padding=1).to(Utils.g_device)
        self.conv2 = nn.Conv2d(in_channel - linearScaling, in_channel - linearScaling, (3, 3), stride=1,
                               padding=1).to(Utils.g_device)

        if not self.last_layer:
            self.conv3 = nn.Conv2d(in_channel - linearScaling, in_channel - linearScaling, (1, 1), stride=1,
                                   padding=0).to(Utils.g_device)
            self.deconv = nn.ConvTranspose2d(in_channel - linearScaling, in_channel - linearScaling, (2, 2), stride=2,
                                             padding=0).to(Utils.g_device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        if not self.last_layer:
            if self.mode == 1:
                x = self.conv3(x)
                x = self.relu(x)
                x = nn.functional.interpolate(x, size=self.outputSize)
            if self.mode == 2:
                x = self.deconv(x)
                x = self.relu(x)
                #x = nn.functional.interpolate(x, size=self.outputSize)
        return x


# flag = G1 or G2
# G1 = With FC in mid
# G2 = With Conv in mid

# mode = 1 or 2
# 1 = With Upsampling (nearest neighbor)
# 2 = With Deconvolution
class Generator(nn.Module):
    def __init__(self, N, imageSize=(3, 100, 100), flag="G1", mode=2, linearScaling=1):
        super(Generator, self).__init__()
        self.flag = flag
        self.relu = nn.ReLU().to(Utils.g_device)
        self.mode = mode
        sampleTensor = torch.zeros(imageSize).to(Utils.g_device)
        sampleTensor = sampleTensor[None]

        # create the N ResidualBlocks and ResidualUpsampleBlocks

        self.encoder = [ResidualBlock(sampleTensor.shape[1], linearScaling, False)]
        out = sampleTensor[0, 0].shape
        sampleTensor = self.encoder[-1](sampleTensor)

        self.decoder = [ResidualBlockUp(sampleTensor.shape[1], linearScaling, False, self.mode, out)]
        for i in range(N - 2):
            layer = ResidualBlock(sampleTensor.shape[1], linearScaling)
            self.encoder.append(layer)
            out = sampleTensor[0, 0].shape
            sampleTensor = self.encoder[-1](sampleTensor)
            layer = ResidualBlockUp(sampleTensor.shape[1], linearScaling, False, self.mode, out)
            self.decoder.append(layer)
        layer = ResidualBlock(sampleTensor.shape[1], linearScaling, True)
        self.encoder.append(layer)
        out = sampleTensor[0, 0].shape
        sampleTensor = self.encoder[-1](sampleTensor)
        layer = ResidualBlockUp(sampleTensor.shape[1], linearScaling, True, self.mode, out)
        self.decoder.append(layer)

        # One Mid-layer
        if flag == "G1":
            size = int(sampleTensor.shape[1] * sampleTensor.shape[2] * sampleTensor.shape[3])
            layer = nn.Linear(size, size).to(Utils.g_device)
        elif flag == "G2":
            layer = nn.Conv2d(sampleTensor.shape[1], sampleTensor.shape[1], (3, 3), stride=1,
                              padding=1).to(Utils.g_device)
        self.midLayer = layer

        # reverse the decoder
        self.decoder.reverse()

    def forward(self, x):
        skips = []
        # encoder
        for l in self.encoder:
            x = l(x)
            skips.append(x)
        # mid layer
        if self.flag == "G1":
            shapeTemp = x.shape
            x = x.view(x.shape[0], -1)
        x = self.relu(self.midLayer(x))
        if self.flag == "G1":
            x = x.view(shapeTemp)

        # decoder
        skips.reverse()
        for l, sx in zip(self.decoder, skips):
            x = l(x + sx)
        return x


class Discriminator(nn.Module):
    def __init__(self, imageSize=(3, 100, 200)):
        super(Discriminator, self).__init__()
        self.imageSize = imageSize
        self.layers = []
        sampleTensor = torch.zeros(imageSize)[None].to(Utils.g_device)
        #4 or 2 are ok ==> 224 neurons or 72 neurons
        while min(sampleTensor.shape[2], sampleTensor.shape[3])>2:
            self.layers.append(nn.Conv2d(sampleTensor.shape[1], sampleTensor.shape[1] + 1,
                                         (3, 3), stride=2, padding=1).to(Utils.g_device))
            sampleTensor = self.layers[-1](sampleTensor)
        size = sampleTensor.shape[1] * sampleTensor.shape[2] * sampleTensor.shape[3]
        self.final = nn.Linear(size, 2).to(Utils.g_device)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        x = x.view(x.shape[0], -1)
        x = self.final(x)
        indices = x.argmax(dim=-1)
        x = x[torch.arange(x.size(0)), indices]
        x = x.clamp(0, 1)
        return x


if __name__ == "__main__":
    inp = torch.zeros((10, 3, 100, 200))
    g = []
    #############################################################
    # flag = G1 or G2
    # G1 = With FC in mid
    # G2 = With Conv in mid
    #############################################################
    # mode = 1 or 2
    # 1 = With Upsampling (nearest neighbor)
    # 2 = With Deconvolution
    #############################################################
    # N, linearScaling, in_channel, imageSize, flag, mode
    #############################################################
    gen = [Generator(3, inp[0].shape, "G2", 2),
           Generator(3, inp[0].shape, "G1", 2),
           Generator(3, inp[0].shape, "G2", 1),
           Generator(3, inp[0].shape, "G1", 1)
           ]
    for g in gen:
        print("")
        out = g(inp)
        print(out.shape)

    # Discriminator testing
    dis = Discriminator(inp[0].shape)
    print("")
    out = dis(inp)
    print(out.shape)
