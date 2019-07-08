########################################################################################################################
# Jan Sieber, 3219317
# Python 3.5
# Deep Vision, University of Heidelberg, Prof. Dr. Björn Ommer
########################################################################################################################

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, linearScaling, last_layer=False):
        super(ResidualBlock, self).__init__()
        self.last_layer = last_layer
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channel, in_channel + linearScaling, (3, 3), stride=1,
                               padding=0)
        self.conv2 = nn.Conv2d(in_channel + linearScaling, in_channel + linearScaling, (3, 3), stride=1,
                               padding=0)
        if not self.last_layer:
            self.conv3 = nn.Conv2d(in_channel + linearScaling, in_channel + linearScaling, (3, 3), stride=2,
                                   padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        if not self.last_layer:
            x = self.conv3(x)
            x = self.relu(x)
        return x


class ResidualBlockUp(nn.Module):
    def __init__(self, in_channel, linearScaling, outputSize, last_layer=False):
        super(ResidualBlockUp, self).__init__()
        self.last_layer = last_layer
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channel, in_channel - linearScaling, (3, 3), stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channel - linearScaling, in_channel - linearScaling, (3, 3), stride=1,
                               padding=1)
        if not self.last_layer:
            self.conv3 = nn.Conv2d(in_channel - linearScaling, in_channel - linearScaling, (1, 1), stride=1,
                                   padding=0)
        self.up = nn.Upsample(size=outputSize)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        if not self.last_layer:
            x = self.conv3(x)
            x = self.relu(x)
        x = self.up(x)
        return x


class Generator(nn.Module):
    def __init__(self, N, linearScaling=1, in_channel=3, imageSize=(3, 100, 100), flag="G1"):
        super(Generator, self).__init__()
        self.flag = flag
        self.relu = nn.ReLU()
        sampleTensor = torch.zeros(imageSize)
        sampleTensor = sampleTensor[None]

        # create the N ResidualBlocks and ResidualUpsampleBlocks
        self.encoder = [ResidualBlock(in_channel, linearScaling)]
        output = sampleTensor[0, 0].shape
        sampleTensor = self.encoder[-1](sampleTensor)
        self.decoder = [ResidualBlockUp(sampleTensor.shape[1], linearScaling, output, True)]

        for i in range(N - 2):
            layer = ResidualBlock(sampleTensor.shape[1], linearScaling)
            self.encoder.append(layer)
            output = sampleTensor[0, 0].shape
            sampleTensor = self.encoder[-1](sampleTensor)
            layer = ResidualBlockUp(sampleTensor.shape[1], linearScaling, output)
            self.decoder.append(layer)
        layer = ResidualBlock(sampleTensor.shape[1], linearScaling, True)
        self.encoder.append(layer)
        output = sampleTensor[0, 0].shape
        sampleTensor = self.encoder[-1](sampleTensor)
        layer = ResidualBlockUp(sampleTensor.shape[1], linearScaling, output)
        self.decoder.append(layer)

        # One FC layer
        if flag == "G1":
            size = int(sampleTensor.shape[1] * sampleTensor.shape[2] * sampleTensor.shape[3])
            layer = nn.Linear(size, size)
        elif flag == "G2":
            layer = nn.Conv2d(sampleTensor.shape[1], sampleTensor.shape[1], (3, 3), stride=1,
                              padding=1)
        self.midLayer = layer

        # reverse the decoder
        self.decoder.reverse()

    def forward(self, x):
        skips = []
        for l in self.encoder:
            x = l(x)
            skips.append(x)
        skips.reverse()
        if self.flag=="G1":
            shapeTemp = x.shape
            x = x.view(x.shape[0], -1)
        x = self.relu(self.midLayer(x))
        if self.flag=="G1":
            x = x.view(shapeTemp)
        for l, sx in zip(self.decoder, skips):
            x = l(x + sx)
        return x


if __name__ == "__main__":
    inp = torch.zeros((10, 3, 100, 100))
    gen = Generator(3, 1, inp.shape[1], inp[0].shape, "G2")
    print("")
    output = gen(inp)
    print(output.shape)
