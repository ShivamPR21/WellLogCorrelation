from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class EncoderDecoder1DCNN(nn.Module):

    def __init__(self,
                 in_channels : int = 2,
                 encoding_len : int = 10,
                 size : int = 100,
                 test : bool = False,
                 norm_layer : Optional[Callable[..., nn.Module]] = nn.BatchNorm1d,
                 activation : Optional[Callable[..., nn.Module]] = nn.SELU) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.encoding_len = encoding_len
        self.size = size
        self.test = test
        self.norm_layer = norm_layer
        self.activation = activation
        if self.activation is None:
            self.activation = nn.ReLU6

        self.encoder, self.enc_size = self.get_encoder()
        self.decoder, self.dec_size = self.get_decoder()
        # self.linears = nn.ModuleList([nn.Linear(self.dec_size, self.size) for i in range(self.in_channels)])

    def conv_layer(self, in_channels : int,
                   out_channels : int,
                   kernel_size : int,
                   stride : int,
                   padding : int,
                   dilation : int,
                   l_in : int,
                   transposed : bool = False,
                   bias:bool = True) -> Tuple[nn.Module, int]:

        if transposed:
            conv_ = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            l_out = (l_in-1)*stride - 2*padding + dilation*(kernel_size-1) + 1

            return conv_, int(l_out)

        conv_ = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        l_out = (l_in + 2*padding - dilation*(kernel_size-1) - 1)/stride+1
        return conv_, int(l_out)


    def get_encoder(self) -> nn.Sequential:
        conv1, l_out = self.conv_layer(self.in_channels, 5, 10, 1, 0, 1, self.size)
        conv2, l_out = self.conv_layer(5, 10, 10, 1, 0, 1, l_out, bias=self.norm_layer is None)
        conv3, l_out = self.conv_layer(10, 10, 10, 1, 0, 1, l_out)
        conv4, l_out = self.conv_layer(10, 5, 10, 1, 0, 1, l_out, bias=self.norm_layer is None)
        conv5, l_out = self.conv_layer(5, 2, 10, 1, 0, 1, l_out)
        # conv6, l_out = self.conv_layer(20, 20, 10, 1, 0, 1, l_out, bias=self.norm_layer is None)
        # conv7, l_out = self.conv_layer(20, 10, 10, 1, 0, 1, l_out)
        # conv8, l_out = self.conv_layer(10, 10, 10, 1, 0, 1, l_out, bias=self.norm_layer is None)
        # conv9, l_out = self.conv_layer(10, 2, 10, 1, 0, 1, l_out)

        # linear = nn.Linear(l_out, self.encoding_len)

        modules : List[nn.Module] = []
        if self.norm_layer is None:
            modules.extend([conv1, self.activation(), conv2, self.activation(),
                            conv3, self.activation(), conv4, self.activation(),
                            conv5])
        else:
            modules.extend([conv1, self.activation(), conv2, self.norm_layer(10), self.activation(),
                            conv3, self.activation(), conv4, self.norm_layer(20), self.activation(),
                            conv5])

        encoder = nn.Sequential(*modules)

        return encoder, l_out

    def get_decoder(self) -> Tuple[nn.Sequential, int]:
        conv1, l_out = self.conv_layer(2, 5, 10, 1, 0, 1, self.enc_size, True)
        conv2, l_out = self.conv_layer(5, 10, 10, 1, 0, 1, l_out, True, bias=self.norm_layer is None)
        conv3, l_out = self.conv_layer(10, 10, 10, 1, 0, 1, l_out, True)
        conv4, l_out = self.conv_layer(10, 5, 10, 1, 0, 1, l_out, True, bias=self.norm_layer is None)
        conv5, l_out = self.conv_layer(5, 2, 5, 1, 0, 1, l_out, False)
        # conv6, l_out = self.conv_layer(20, 20, 10, 1, 0, 1, l_out, False, bias=True)
        # conv7, l_out = self.conv_layer(20, 10, 5, 1, 0, 1, l_out, True)
        # conv8, l_out = self.conv_layer(10, 10, 5, 1, 0, 1, l_out, True, bias=self.norm_layer is None)
        # conv9, l_out = self.conv_layer(10, 2, 5, 1, 0, 1, l_out, False)

        modules : List[nn.Module] = []
        if self.norm_layer is None:
            modules.extend([conv1, self.activation(), conv2, self.activation(),
                            conv3, self.activation(), conv4, self.activation(),
                            conv5])
        else:
            modules.extend([conv1, self.activation(), conv2, self.norm_layer(10), self.activation(),
                            conv3, self.activation(), conv4, self.norm_layer(20), self.activation(),
                            conv5])

        decoder = nn.Sequential(*modules)

        return decoder, l_out

    def forward(self, x): # [B, 2, self.size]
        encoding = self.encoder(x) # [B, 2, self.size] -> [B, 2, self.enc_size]
        enc_feature1, enc_feature2 = encoding.split(1, dim=1)
        enc_feature1, enc_feature2 = interpolate(enc_feature1, self.encoding_len), interpolate(enc_feature2, self.encoding_len)

        encoding = torch.cat((enc_feature1, enc_feature2), dim=1)

        if self.test:
            return encoding

        decoded: torch.Tensor = self.decoder(encoding) # [B, 1, self.enc_size] -> [B, 2, self.dec_size]
        feature1, feature2 = decoded.split(1, dim=1)
        feature1, feature2 = interpolate(feature1, self.size, mode='linear'), interpolate(feature2, self.size, mode='linear')

        output = torch.cat((feature1, feature2), dim=1)

        output = output / torch.norm(output, dim=2, keepdim=True)+0.00000001
        # output = output / torch.norm(output, dim=1, keepdim=True)+0.00000001

        return output
