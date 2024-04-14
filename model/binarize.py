from torch.nn import Conv2d, Linear
import torch.nn.functional as F 
from torch.autograd import Function
from math import sqrt

class Binarize(Function):

    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    

class BinaryLinear(Linear):

    def forward(self, input):
        binary_weight = Binarize.apply(self.weight)
        if self.bias is None:
            return F.linear(input, binary_weight)
        else:
            return F.linear(input, binary_weight, self.bias)

    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv
        

class BinaryConv2d(Conv2d):

    def forward(self, input):
        bw = Binarize.apply(self.weight)
        return F.conv2d(input, bw, self.bias, self.stride,
                               self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        # Glorot initialization
        in_features = self.in_channels
        out_features = self.out_channels
        
        for k in self.kernel_size:
            in_features *= k
            out_features *= k
            
        stdv = sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv
        
        
# def forward(self, input):
#     input = input.float()
#     batch_size, time_bins, polarities, height, width = input.size()
    
#     input = input.view(batch_size * time_bins, polarities, height, width)
    
#     for layer in self.layers:
#         if isinstance(layer, Leaky):
#             layer.init_leaky()
    
#     spike_rec = []
#     mem_rec = []