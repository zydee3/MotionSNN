from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Flatten, Linear, Dropout, init, F, Function

from snntorch import Leaky
from snntorch.surrogate import atan

DEFAULT_BSNN_CONFIG_VALUES = {
    'num_blocks': 2,        
    'num_classes': 10,
    'img_dimension': 28,
    'base_width': 8,
    'base_factor': 2,
    'pool_kernel_size': 2,
    'pool_stride_size': 2,
    'conv_kernel_size': 5,
    'conv_stride_size': 1,
    'conv_padding_size': 0,
    'conv_dilation_size': 1,
}

def prepare_config(config):
    if config is None:
        return DEFAULT_BSNN_CONFIG_VALUES.copy()
    
    prepared_config = config.copy()
    
    for key, value in DEFAULT_BSNN_CONFIG_VALUES.items():
        prepared_config.setdefault(key, value)
        
    return prepared_config


class Binarize(Function):
  @staticmethod
  def forward(weight_ref, inpt):
    return inpt.sign().clamp(min=-1)

  @staticmethod
  def backward(weight_ref, gradient_out):
    gradient_in = gradient_out.clone()
    return gradient_in


class BinaryConv2d(Conv2d):
  def __init__(self, *kargs, **kwargs):
    super(BinaryConv2d, self).__init__(*kargs, **kwargs)

  def forward(self, inpt):
    binarized_weights = Binarize.apply(self.weight)
    return F.conv2d(inpt, binarized_weights)

  def reset_params(self):
    init.xavier_normal_(self.weight)
    if self.bias is not None:
      init.constant(self.bias, 0)


class BinaryLinear(Linear):
  def __init__(self, *kargs, **kwargs):
    super(BinaryLinear, self).__init__(*kargs, **kwargs)

    def forward(self, inpt):
        bin_weights = Binarize.apply(self.weight)
        if self.bias is None:
            return F.linear(inpt, bin_weights)
        else:
            return F.linear(inpt, bin_weights, self.bias)

    def reset_parameters(self):
        init.xavier_normal_(self.weight)
        if self.bias is not None:
            init.constant_(self.bias, 0)
            

class BSNN(Module):
    def __init__(self, config=None):
        super().__init__()
        
        self.config = prepare_config(config)
        self.layers = Sequential()
        
        self._build()
    
    
    def _calc_pool_feat_map_out_size(self, in_feat_map_size):
        kernel_size = self.config['pool_kernel_size']
        stride_size = self.config['pool_stride_size']
        return ((in_feat_map_size - kernel_size) // stride_size) + 1


    def _calc_pool_channel_in_size(self, block_idx):
        if block_idx == 0:
            return 1
        
        base_width = self.config['base_width']
        base_factor = self.config['base_factor']
        return base_width * (base_factor ** (block_idx - 1))

    
    def _calc_pool_channel_out_size(self, block_idx):
        return self._calc_pool_channel_in_size(block_idx + 1)
                   
                   
    def _calc_conv_feat_map_out_size(self, in_feat_map_size):
        padding_size = self.config['conv_padding_size']
        kernel_size = self.config['conv_kernel_size']
        stride_size = self.config['conv_stride_size']
        dilation_size = self.config['conv_dilation_size']
        return ((in_feat_map_size + 2 * padding_size - dilation_size * (kernel_size - 1) - 1) // stride_size) + 1

    
    def _calc_feat_map_size(self, num_blocks, in_feat_map_size):
        for _ in range(num_blocks):
            in_feat_map_size = self._calc_conv_feat_map_out_size(in_feat_map_size)
            in_feat_map_size = self._calc_pool_feat_map_out_size(in_feat_map_size)
            
        return in_feat_map_size
    
    
    def _add_block(self, block_idx, activation_threshold=0.5):
        conv_in_size = self._calc_pool_channel_in_size(block_idx)
        conv_out_size = self._calc_pool_channel_out_size(block_idx)
        
        self.layers.add_module(f'conv{block_idx}', BinaryConv2d(
            in_channels=conv_in_size,
            out_channels=conv_out_size,
            kernel_size=self.config['conv_kernel_size'],
            stride=self.config['conv_stride_size'],
            padding=self.config['conv_padding_size'],
            dilation=self.config['conv_dilation_size'],
        ))
        
        self.layers.add_module(f'pool{block_idx}', MaxPool2d(
            kernel_size=self.config['pool_kernel_size'],
            stride=self.config['pool_stride_size'],
        ))
        
        self.layers.add_module(f'relu{block_idx}', Leaky(
            beta=activation_threshold, 
            spike_grad=atan,
            init_hidden=True
        ))
        
        return conv_out_size


    def _add_output_layer(self, in_channel_size, activation_threshold=0.4, dropout_rate=0.25):        
        feat_map_size = self._calc_feat_map_size(
            num_blocks=self.config['num_blocks'], 
            in_feat_map_size=self.config['img_dimension'],
        )
        
        self.layers.add_module('flatten', Flatten())
        self.layers.add_module('output', BinaryLinear(
            in_features=in_channel_size * (feat_map_size ** 2),
            out_features=self.config['num_classes'],
        ))
        
        self.layers.add_module('dropout', Dropout(dropout_rate))
        
        self.layers.add_module('leaky_out', Leaky(
            beta=activation_threshold, 
            spike_grad=atan,
            init_hidden=True,
            output=True
        ))
    
    
    def _build(self):
        for block_idx in range(self.config['num_blocks']):
            out_size = self._add_block(block_idx)
            
            if block_idx == self.config['num_blocks'] - 1:
                self._add_output_layer(out_size)
                

    def forward(self, input):
        input = input.float()
        batch_size, time_bins, polarities, height, width = input.size()
        
        input = input.view(batch_size * time_bins, polarities, height, width)
        
        for layer in self.layers:
            if isinstance(layer, Leaky):
                layer.init_leaky()
        
        spike_rec = []
        mem_rec = []
        
        