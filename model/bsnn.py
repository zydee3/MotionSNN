from torch.nn import Module, Sequential, Flatten, Dropout, BatchNorm2d

from snntorch import Leaky
from snntorch.surrogate import atan, fast_sigmoid

from model.binarize import BinaryLinear, BinaryConv2d
from model.config import prepare_config, DEFAULT_BSNN_CONFIG_VALUES
  

class BSNN(Module):
    def __init__(self, config=None):
        super().__init__()
        
        self.config = prepare_config(config, DEFAULT_BSNN_CONFIG_VALUES)
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
    
    
    def _add_block(self, block_idx):
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
        
        self.layers.add_module(f'pool{block_idx}', BatchNorm2d(
            kernel_size=self.config['pool_kernel_size'],
            stride=self.config['pool_stride_size'],
        ))
        
        self.layers.add_module(f'relu{block_idx}', Leaky(
            beta=self.config['beta_value'], 
            spike_grad=fast_sigmoid(slope=self.config['slope']),
            threshold=self.config['thresholds'][block_idx],
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
        print(input.size(0))
        # for step in range(input.size(0)):