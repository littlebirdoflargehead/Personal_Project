# modify con2d function to use same padding
# code referd to @famssa in 'https://github.com/pytorch/pytorch/issues/3867'
# and tensorflow source code

import torch.utils.data
from torch.nn import functional as F

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.conv import _ConvNd,_ConvTransposeMixin
from torch.nn.modules.utils import _single, _pair, _triple



class Conv2d_same(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d_same, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)



# custom con2d, because pytorch don't have "padding='same'" option.
def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):

    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows -
                  input_rows)
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)



class ConvTranspose2d_same(_ConvTransposeMixin, _ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(ConvTranspose2d_same, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, input, output_size=None):
        # type: (Tensor, Optional[List[int]]) -> Tensor
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)

        rows_padding = self.kernel_size[0]-self.stride[0]
        rows_odd =  rows_padding % 2 != 0
        rows_padding //= 2
        cols_padding = self.kernel_size[1]-self.stride[1]
        cols_odd =  cols_padding % 2 != 0
        cols_padding //= 2
        output = F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation
        )
        output_size = output.shape

        return output[:,:,int(rows_odd)+rows_padding:output_size[2]-rows_padding,
               int(cols_odd)+cols_padding:output_size[3]-cols_padding]
