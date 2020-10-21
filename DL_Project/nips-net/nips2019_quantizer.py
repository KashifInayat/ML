import torch as t
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

 

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class NIPSQuantizer(t.nn.Module):
    def __init__(self, bit, is_gaussian=False,  is_activation=False):
        super(NIPSQuantizer,self).__init__()

        self.bit = bit
        self.is_gaussian = is_gaussian
        self.is_activation = is_activation
        
        if self.is_gaussian:
            # Gaussian        
            self.coff = {1: 1.24, 2: 1.71, 3: 2.15, 4: 2.55, 5: 2.93, 6: 3.28, 7: 3.61, 8: 3.92}
            self.coff_positive = {1: 1.71, 2: 2.15, 3: 2.55, 4: 2.93, 5: 3.28, 6: 3.61, 7: 3.92, 8: 4.2}
        else:
            # Laplace
            self.coff = {0: 1.05, 1: 1.86, 2: 2.83, 3: 3.89, 4: 5.03, 5: 6.2, 6: 7.41, 7: 8.64, 8: 9.89}
            self.coff_positive = {0: 1.86, 1: 2.83, 2: 3.89, 3: 5.02, 4: 6.2, 5: 7.41, 6: 8.64, 7: 9.89, 8: 11.16}




    def forward(self, x):
        b = ((x.var() / 2) ** 0.5)
        if self.is_activation:
            alpha = self.coff_positive[self.bit] * b
            _min = 0
            _max = alpha
            step_size = alpha / 2 ** self.bit

            #x_clamp = 0.5*(torch.abs(x) - torch.abs(x-alpha)+alpha) 
			x_clamp = torch.min(torch.max(x, _min),  _max)
            x_q = round_pass(x_clamp / step_size) * step_size


        else:
            alpha = self.coff[self.bit] * b
            _min = -1.0 * alpha
            _max = alpha

            step_size = alpha / 2 ** self.bit
            #x_clamp = -1.0 * 0.5*(torch.abs(x) - torch.abs(x+alpha)+alpha) + 0.5*(torch.abs(x) - torch.abs(x-alpha) + alpha)
			x_clamp = torch.min(torch.max(x, _min),  _max)


            x_q = round_pass(x_clamp / step_size) * step_size
        # g = 1.0 / math.sqrt(x.numel() * self.Qp)
        # _alpha = grad_scale(self.alpha, g)
		
		# because we use symetric quantization, so we don't include 0-point value in the results		
		x_q[x_q==0] = step_size
        return x_q

    def extra_repr(self):
        return "bit=%s, is_activation=%s" % (self.bit, self.is_activation)


class NIPS2019_QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, bit=4):


        super(NIPS2019_QConv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.quan_w = NIPSQuantizer(bit=bit, is_activation=False, is_gaussian=False)
        self.quan_a = NIPSQuantizer(bit=bit, is_activation=True, is_gaussian=False)
        self.bit = bit

    def forward(self, x):
        if self.bit == 32:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(self.quan_a(x), self.quan_w(self.weight), self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

class NIPS2019_QInputConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, bit=4):


        super(NIPS2019_QInputConv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.quan_w = NIPSQuantizer(bit=bit, is_activation=False, is_gaussian=False)
        self.quan_a = NIPSQuantizer(bit=bit, is_activation=False, is_gaussian=False)
        self.bit = bit

    def forward(self, x):
        if self.bit == 32:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(self.quan_a(x), self.quan_w(self.weight), self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

class NIPS2019_QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bit=4):
        super(NIPS2019_QLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        
        self.bit = bit
        self.quan_w = NIPSQuantizer(bit=bit, is_activation=False, is_gaussian=False)
        self.quan_a = NIPSQuantizer(bit=bit, is_activation=True, is_gaussian=False)

    def forward(self, x):
        if self.bit == 32:
            return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(self.quan_a(x), self.quan_w(self.weight), self.bias)


