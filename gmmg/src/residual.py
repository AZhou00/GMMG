r"""
Residual Block.
"""
from typing import Any

import jax
import jax.numpy as jnp

import flax.linen as nn

# residual block for training deep networks
class ResBlock(nn.Module):
    """Basic Residual Block."""

    filters: int # how many convolutional filters to use = number of output channels
    norm_fn: Any # group norm, layer norm, etc. (batch norm not implemented)
    conv_fn: Any # vanilla convolution, flax.linen.Conv.
    dtype: int = jnp.float32
    activation_fn: Any = nn.relu # relu or swish
    
    # if True, use 3x3 convolution for shortcut, else use 1x1 convolution
    use_conv_shortcut: bool = False

    @nn.compact
    def __call__(self, x):
        """
        1. (normalization -> activation -> convolution) * 2
        2. add residual connection with convolutional shortcut

        input dim: (batch, spatial_dims…, features)
        output dim: (batch, spatial_dims…, filters)

        convolution padding: default is 'SAME', see https://flax.readthedocs.io/en/v0.5.3/_autosummary/flax.linen.Conv.html

        Since the input and output dimensions are in general different, we need to use a convolutional shortcut.
            if use_conv_shortcut:
                use 3x3 convolution for shortcut
            else:
                use 1x1 convolution for shortcut (just to change the number of channels)
        """
        input_dim = x.shape[-1]
        residual = x 
        x = self.norm_fn()(x)
        x = self.activation_fn(x)
        x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)
        x = self.norm_fn()(x)
        x = self.activation_fn(x)
        x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)

        if input_dim != self.filters:
            if self.use_conv_shortcut:
                residual = self.conv_fn(
                    self.filters, kernel_size=(3, 3), use_bias=False
                )(x)
            else:
                residual = self.conv_fn(
                    self.filters, kernel_size=(1, 1), use_bias=False
                )(x)
        return x + residual