from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn


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
        self.linears = nn.ModuleList([nn.Linear(self.dec_size, self.size) for i in range(self.in_channels)])

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
        conv1, l_out = self.conv_layer(self.in_channels, 5, 10, 1, 9, 1, self.size)
        conv2, l_out = self.conv_layer(5, 10, 10, 1, 9, 1, l_out, bias=self.norm_layer is None)
        conv3, l_out = self.conv_layer(10, 10, 10, 1, 9, 1, l_out)
        conv4, l_out = self.conv_layer(10, 10, 10, 1, 9, 1, l_out, bias=self.norm_layer is None)
        conv5, l_out = self.conv_layer(10, 1, 10, 1, 9, 1, l_out)

        linear = nn.Linear(l_out, self.encoding_len)

        modules : List[nn.Module] = []
        if self.norm_layer is None:
            modules.extend([conv1, self.activation(), conv2, self.activation(),
                            conv3, self.activation(), conv4, self.activation(),
                            conv5, self.activation(), linear])
        else:
            modules.extend([conv1, self.activation(), conv2, self.norm_layer(10), self.activation(),
                            conv3, self.activation(), conv4, self.norm_layer(10), self.activation(),
                            conv5, self.activation(), linear])

        encoder = nn.Sequential(*modules)

        return encoder, self.encoding_len

    def get_decoder(self) -> Tuple[nn.Sequential, int]:
        conv1, l_out = self.conv_layer(1, 5, 10, 1, 0, 1, self.encoding_len, True)
        conv2, l_out = self.conv_layer(5, 10, 10, 1, 0, 1, l_out, True, bias=self.norm_layer is None)
        conv3, l_out = self.conv_layer(10, 10, 10, 1, 0, 1, l_out, True)
        conv4, l_out = self.conv_layer(10, 5, 10, 1, 0, 1, l_out, True, bias=self.norm_layer is None)
        conv5, l_out = self.conv_layer(5, 5, 10, 1, 0, 1, l_out, True)
        conv6, l_out = self.conv_layer(5, 2, 10, 1, 0, 1, l_out, True)

        modules : List[nn.Module] = []
        if self.norm_layer is None:
            modules.extend([conv1, self.activation(), conv2, self.activation(),
                            conv3, self.activation(), conv4, self.activation(),
                            conv5, self.activation(), conv6, self.activation()])
        else:
            modules.extend([conv1, self.activation(), conv2, self.norm_layer(10), self.activation(),
                            conv3, self.activation(), conv4, self.norm_layer(5), self.activation(),
                            conv5, self.activation(), conv6, self.activation()])

        decoder = nn.Sequential(*modules)

        return decoder, l_out

    def forward(self, x): # [B, 2, self.size]
        b = x.shape[0]
        encoding = self.encoder(x) # [B, 2, self.size] -> [B, 1, self.enc_size]

        if self.test:
            return encoding

        decoded = self.decoder(encoding) # [B, 1, self.enc_size] -> [B, 2, self.dec_size]

        features = torch.permute(decoded, (1, 0, 2))
        out_features = (self.linears[i](features[i, :, :]) for i in range(self.in_channels))

        output = torch.cat(tuple(out_features), dim=0).view(self.in_channels, b, self.size)
        output = torch.permute(output, (1, 0, 2))

        output = output / torch.norm(output, dim=1, keepdim=True)+0.0000001
        output = output / torch.norm(output, dim=1, keepdim=True)+0.0000001

        return output
