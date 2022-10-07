import math
from typing import Tuple

import torch
import torch.nn as nn


@torch.jit.script
def Pixelshuffle(x: torch.Tensor, factor_hw: Tuple[int, int]):
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC // (pH * pW), iH * pH, iW * pW
    y = y.reshape(B, oC, pH, pW, iH, iW)
    y = y.permute(0, 1, 4, 2, 5, 3)  # B, oC, iH, pH, iW, pW
    y = y.reshape(B, oC, oH, oW)
    return y


@torch.jit.script
def Pixelshuffle_invert(x: torch.Tensor, factor_hw: Tuple[int, int]):
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC * (pH * pW), iH // pH, iW // pW
    y = y.reshape(B, iC, oH, pH, oW, pW)
    y = y.permute(0, 1, 3, 5, 2, 4)  # B, iC, pH, pW, oH, oW
    y = y.reshape(B, oC, oH, oW)
    return y


class Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, strdie=1, padding=2):
        super(Basic_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kszie = ksize
        self.padding = padding
        self.conv_layer = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel_size=(ksize, ksize), stride=1, padding=(padding, padding))
        self.act_layer = nn.LeakyReLU()

    def forward(self, x):
        return self.act_layer(self.conv_layer(x))


class Sub_pixel(nn.Module):
    def __init__(self, in_channels, ksize=3, scaling_factor=2):
        super(Sub_pixel, self).__init__()
        self.out_channesl = in_channels * (scaling_factor ** 2)
        self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=self.out_channesl, kernel_size=(ksize, ksize),padding=1)
        self.pixelShuffle_layer = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.last_layer = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=(ksize, ksize),padding=1)

    def forward(self, x):
        # output shape: H x W x (C x r**2)
        x = self.conv_layer(x)

        # Sub-Pixel Convolution Layer -->  PixelShuffle
        # rearranges: H x W x (C x r**2) => rH x rW x C
        x = self.pixelShuffle_layer(x)
        x = self.last_layer(x)
        return x


class ESPCN(nn.Module):
    def __init__(self, num_channels, scaling_factor):
        super(ESPCN, self).__init__()
        self.scaling_factor = scaling_factor
        self.conv_layer_1 = Basic_Block(in_channels=num_channels, out_channels=64, ksize=5, padding=2)
        self.conv_layer_2 = Basic_Block(in_channels=64, out_channels=32, ksize=3, padding=1)

        self.pixelShuffle_last_layer = Sub_pixel(in_channels=32, ksize=3, scaling_factor=scaling_factor)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0,
                                    std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        # x = Pixelshuffle( x, self.scaling_factor)
        x = self.pixelShuffle_last_layer(x)
        return x


'''
if __name__ == '__main__':
    import torch.nn.functional as F
    print('Check function correct')
    print()

    for s in [1, 2, 4, 8, 16]:
        print('Checking scale {}'.format(s))
        x = torch.rand(5, 256, 128, 128)   # BCHW

        y1 = F.pixel_shuffle(x, s)
        print("before shape " , x.shape , " at s : " , s )
        print("after shape : ",y1.shape)
        y2 = pixelshuffle(x, (s, s))

        assert torch.allclose(y1, y2)
        print('pixelshuffle works correctly.')

        rev_x = pixelshuffle_invert(y1, (s, s))

        assert torch.allclose(x, rev_x)
        print('pixelshuffle_invert works correctly.')
        print()
'''

if __name__ == '__main__':
    Image = torch.rand(size=(1, 1, 224, 224))
    print("Input shape: ", Image.shape)

    model = ESPCN(num_channels=1, scaling_factor=3)
    print(f"\n{model}\n")

    output = model(Image)
    print(f"output shape: {output.shape}")

    '''
    Input shape:  torch.Size([1, 1, 224, 224])

    ESPCN(
      (conv_layer_1): Basic_Block(
        (conv_layer): Conv2d(1, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (act_layer): Tanh()
      )
      (conv_layer_2): Basic_Block(
        (conv_layer): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act_layer): Tanh()
      )
      (pixelShuffle_layer): Sub_pixel(
        (conv_layer): Conv2d(32, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (pixelShuffle_layer): PixelShuffle(upscale_factor=3)
      )
    )
    
    output shape: torch.Size([1, 32, 672, 672])
    '''
