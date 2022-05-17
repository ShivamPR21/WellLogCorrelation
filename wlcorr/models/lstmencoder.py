from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import interpolate


def conv_layer(in_channels : int,
                out_channels : int,
                kernel_size : int,
                stride : int,
                padding : int,
                dilation : int,
                l_in : int,
                transposed : bool = False,
                bias:bool = True) -> Tuple[nn.Module, List[int]]:

    if transposed:
        conv_ = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        l_out = int((l_in-1)*stride - 2*padding + dilation*(kernel_size-1) + 1)
        out_size = [l_out, out_channels]

        return conv_, out_size

    conv_ = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
    l_out = int((l_in + 2*padding - dilation*(kernel_size-1) - 1)/stride+1)
    out_size = [l_out, out_channels]

    return conv_, out_size

def lstm_layer(in_channels: int,
                out_channels: int,
                num_layers: int,
                in_size: int)-> Tuple[nn.Module, List[int]]:
    lstm = nn.LSTM(in_channels, out_channels, num_layers, batch_first=True)
    out_size = [in_size, out_channels]

    return lstm, out_size

class Encoder1DCNN_LSTM(nn.Module):

    def __init__(self, in_channels: int=2,
                 in_size: int=100,
                 norm_layer : Optional[Callable[..., nn.Module]] = nn.BatchNorm1d,
                 activation : Optional[Callable[..., nn.Module]] = nn.SELU) -> None:
        super().__init__()
        self.activation = activation()

        self.cnn1, self.cnn1_size = conv_layer(in_channels, 5, 5, 1, 0, 1, in_size)
        self.lstm1, self.lstm1_size = lstm_layer(self.cnn1_size[1], 5, num_layers=1, in_size=self.cnn1_size[0])

        self.cnn2, self.cnn2_size = conv_layer(self.lstm1_size[1], 10, 5, 1, 0, 1, self.lstm1_size[0])
        self.lstm2, self.lstm2_size = lstm_layer(self.cnn2_size[1], 10, num_layers=1, in_size=self.cnn2_size[0])

        self.cnn3, self.cnn3_size = conv_layer(self.lstm2_size[1], 20, 5, 1, 0, 1, self.lstm2_size[0])
        self.lstm3, self.lstm3_size = lstm_layer(self.cnn3_size[1], 20, num_layers=1, in_size=self.cnn3_size[0])

        self.cnn4, self.cnn4_size = conv_layer(self.lstm3_size[1], 20, 5, 1, 0, 1, self.lstm3_size[0])
        self.lstm4, self.lstm4_size = lstm_layer(self.cnn4_size[1], 20, num_layers=1, in_size=self.cnn4_size[0])

        self.cnn5, self.cnn5_size = conv_layer(self.lstm4_size[1], 20, 5, 1, 0, 1, self.lstm4_size[0])
        self.lstm5, self.lstm5_size = lstm_layer(self.cnn5_size[1], 20, num_layers=1, in_size=self.cnn5_size[0])

        self.conv_last, self.size_last = conv_layer(self.lstm5_size[1], 10, 5, 1, 0, 1, self.lstm5_size[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> [B, n_ch, l]
        B, in_ch, l = x.shape

        x = self.cnn1(x).transpose(2, 1) # [B, l_out, n_ch_out]
        x = self.activation(x)
        hn, cn = torch.rand(1, B, self.lstm1_size[1], device=x.device), torch.rand(1, B, self.lstm1_size[1], device=x.device)
        x, (hn, cn) = self.lstm1(x, (hn, cn)) # [B, n_ch_out, l_out]
        x = x.transpose(2, 1)

        x = self.cnn2(x).transpose(2, 1) # [B, l_out, n_ch_out]
        x = self.activation(x)
        hn, cn = torch.rand(1, B, self.lstm2_size[1], device=x.device), torch.rand(1, B, self.lstm2_size[1], device=x.device)
        x, (hn, cn) = self.lstm2(x, (hn, cn)) # [B, n_ch_out, l_out]
        x = x.transpose(2, 1)

        x = self.cnn3(x).transpose(2, 1) # [B, l_out, n_ch_out]
        x = self.activation(x)
        hn, cn = torch.rand(1, B, self.lstm3_size[1], device=x.device), torch.rand(1, B, self.lstm3_size[1], device=x.device)
        x, (hn, cn) = self.lstm3(x, (hn, cn)) # [B, n_ch_out, l_out]
        x = x.transpose(2, 1)

        x = self.cnn4(x).transpose(2, 1) # [B, l_out, n_ch_out]
        x = self.activation(x)
        hn, cn = torch.rand(1, B, self.lstm4_size[1], device=x.device), torch.rand(1, B, self.lstm4_size[1], device=x.device)
        x, (hn, cn) = self.lstm4(x, (hn, cn)) # [B, n_ch_out, l_out]
        x = x.transpose(2, 1)

        x = self.cnn5(x).transpose(2, 1) # [B, l_out, n_ch_out]
        x = self.activation(x)
        hn, cn = torch.rand(1, B, self.lstm5_size[1], device=x.device), torch.rand(1, B, self.lstm5_size[1], device=x.device)
        x, (hn, cn) = self.lstm5(x, (hn, cn)) # [B, n_ch_out, l_out]
        x = x.transpose(2, 1)

        x = self.conv_last(x) # [B, n_ch_out, l_out]

        return x

    def out_size(self):
        return self.size_last


class Decoder1DCNN_LSTM(nn.Module):

    def __init__(self, in_channels: int,
                 in_size: int,
                 norm_layer : Optional[Callable[..., nn.Module]] = nn.BatchNorm1d,
                 activation : Optional[Callable[..., nn.Module]] = nn.SELU) -> None:
        super().__init__()
        self.activation = activation()
        # self.encoding_activation = nn.Sigmoid()

        self.cnn1, self.cnn1_size = conv_layer(in_channels, 20, 5, 1, 0, 1, in_size, True)
        self.lstm1, self.lstm1_size = lstm_layer(self.cnn1_size[1], 5, num_layers=1, in_size=self.cnn1_size[0])

        self.cnn2, self.cnn2_size = conv_layer(self.lstm1_size[1], 10, 5, 1, 0, 1, self.lstm1_size[0], True)
        self.lstm2, self.lstm2_size = lstm_layer(self.cnn2_size[1], 10, num_layers=1, in_size=self.cnn2_size[0])

        self.cnn3, self.cnn3_size = conv_layer(self.lstm2_size[1], 5, 5, 1, 0, 1, self.lstm2_size[0], True)
        self.lstm3, self.lstm3_size = lstm_layer(self.cnn3_size[1], 5, num_layers=1, in_size=self.cnn3_size[0])

        self.cnn4, self.cnn4_size = conv_layer(self.lstm3_size[1], 5, 5, 1, 0, 1, self.lstm3_size[0], True)
        self.lstm4, self.lstm4_size = lstm_layer(self.cnn4_size[1], 5, num_layers=1, in_size=self.cnn4_size[0])

        self.cnn5, self.cnn5_size = conv_layer(self.lstm4_size[1], 5, 5, 1, 0, 1, self.lstm4_size[0], True)
        self.lstm5, self.lstm5_size = lstm_layer(self.cnn5_size[1], 5, num_layers=1, in_size=self.cnn5_size[0])

        self.conv_last, self.size_last = conv_layer(self.lstm5_size[1], 1, 3, 1, 0, 1, self.lstm5_size[0], False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> [B, n_ch, l]
        B, in_ch, l = x.shape

        x = self.cnn1(x).transpose(2, 1) # [B, l_out, n_ch_out]
        x = self.activation(x)
        hn, cn = torch.rand(1, B, self.lstm1_size[1], device=x.device), torch.rand(1, B, self.lstm1_size[1], device=x.device)
        x, (hn, cn) = self.lstm1(x, (hn, cn)) # [B, n_ch_out, l_out]
        x = x.transpose(2, 1)

        x = self.cnn2(x).transpose(2, 1) # [B, l_out, n_ch_out]
        x = self.activation(x)
        hn, cn = torch.rand(1, B, self.lstm2_size[1], device=x.device), torch.rand(1, B, self.lstm2_size[1], device=x.device)
        x, (hn, cn) = self.lstm2(x, (hn, cn)) # [B, n_ch_out, l_out]
        x = x.transpose(2, 1)

        x = self.cnn3(x).transpose(2, 1) # [B, l_out, n_ch_out]
        x = self.activation(x)
        hn, cn = torch.rand(1, B, self.lstm3_size[1], device=x.device), torch.rand(1, B, self.lstm3_size[1], device=x.device)
        x, (hn, cn) = self.lstm3(x, (hn, cn)) # [B, n_ch_out, l_out]
        x = x.transpose(2, 1)

        x = self.cnn4(x).transpose(2, 1) # [B, l_out, n_ch_out]
        x = self.activation(x)
        hn, cn = torch.rand(1, B, self.lstm4_size[1], device=x.device), torch.rand(1, B, self.lstm4_size[1], device=x.device)
        x, (hn, cn) = self.lstm4(x, (hn, cn)) # [B, n_ch_out, l_out]
        x = x.transpose(2, 1)

        x = self.cnn5(x).transpose(2, 1) # [B, l_out, n_ch_out]
        x = self.activation(x)
        hn, cn = torch.rand(1, B, self.lstm5_size[1], device=x.device), torch.rand(1, B, self.lstm5_size[1], device=x.device)
        x, (hn, cn) = self.lstm5(x, (hn, cn)) # [B, n_ch_out, l_out]
        x = x.transpose(2, 1)

        x = self.conv_last(x) # [B, n_ch_out, l_out]
        # x = self.encoding_activation(x)

        return x

    def out_size(self):
        return self.size_last

class EncoderDecoder1DCNN_LSTM(nn.Module):

    def __init__(self,
                 in_channels : int = 2,
                 in_size : int = 100,
                 test : bool = False,
                 norm_layer : Optional[Callable[..., nn.Module]] = nn.BatchNorm1d,
                 activation : Optional[Callable[..., nn.Module]] = nn.SELU) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.in_size = in_size
        self.test = test
        self.norm_layer = norm_layer
        self.activation = activation
        if self.activation is None:
            self.activation = nn.ReLU6

        self.encoder = Encoder1DCNN_LSTM(self.in_channels, self.in_size, self.norm_layer, self.activation)
        self.encoding_size = self.encoder.out_size()

        print(self.encoding_size)

        self.decoder = Decoder1DCNN_LSTM(self.encoding_size[1], self.encoding_size[0], self.norm_layer, self.activation)
        self.decoding_size = self.decoder.out_size()

    def forward(self, x): # [B, 2, self.size]
        encoding = self.encoder(x) # [B, 2, self.size] -> [B, 2, self.enc_size]
        # enc_feature1, enc_feature2 = encoding.split(1, dim=1)
        # enc_feature1, enc_feature2 = interpolate(enc_feature1, self.encoding_len), interpolate(enc_feature2, self.encoding_len)

        # encoding = torch.cat((enc_feature1, enc_feature2), dim=1)

        if self.test:
            return encoding

        output: torch.Tensor = self.decoder(encoding) # [B, 1, self.enc_size] -> [B, 2, self.dec_size]
        output = interpolate(output, self.in_size, mode='linear')
        # feature1, feature2 = decoded.split(1, dim=1)
        # feature1, feature2 = interpolate(feature1, self.in_size, mode='linear'), interpolate(feature2, self.in_size, mode='linear')

        # output = torch.cat((feature1, feature2), dim=1)

        output = output / torch.norm(output, dim=2, keepdim=True)+0.00000001
        # output = output / torch.norm(output, dim=1, keepdim=True)+0.00000001

        return output
