r"""MaskGIT Tokenizer based on VQGAN.

This tokenizer is a reimplementation of VQGAN [https://arxiv.org/abs/2012.09841]
with several modifications. The non-local layer are removed from VQGAN for
faster speed.
"""
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn

import ml_collections

from src import layer
from src.residual import ResBlock

class Encoder(nn.Module):
    """Encoder Blocks."""

    config: ml_collections.ConfigDict
    train: bool
    dtype: int = jnp.float32

    def setup(self):
        self.filters = self.config.vqvae.filters
        self.num_res_blocks = self.config.vqvae.num_res_blocks
        self.channel_multipliers = self.config.vqvae.channel_multipliers
        self.embedding_dim = self.config.vqvae.embedding_dim
        # convolution downsample or average pooling downsample in encoder
        self.conv_downsample = self.config.vqvae.conv_downsample

        # type of convolutional layer
        if self.config.vqvae.conv_fn == "conv":
            self.conv_fn = nn.Conv
        else:
            raise NotImplementedError
        
        # normalization layer in the residual block
        self.norm_type = self.config.vqvae.norm_type

        # type of activation function
        if self.config.vqvae.activation_fn == "relu":
            self.activation_fn = nn.relu
        elif self.config.vqvae.activation_fn == "swish":
            self.activation_fn = nn.swish
        else:
            raise NotImplementedError

    @nn.compact
    def __call__(self, x):
        """
        input data with dimensions (batch, spatial_dims…, features).
        This is the channels-last convention, i.e. NHWC for a 2d convolution and NDHWC for a 3D convolution.
        Note: this is different from the input convention used by lax.conv_general_dilated,
        which puts the spatial dimensions last.

        --------------------- block 1 (less features)
        residual blocks * num_res_blocks
        downsample
        --------------------- end block 1
        
        --------------------- block 2 (more features)
        residual blocks * num_res_blocks
        downsample
        --------------------- end block 2
        
        ...
        
        --------------------- final block
        residual blocks * num_res_blocks
        normalization -> activation -> 1x1 convolution to embedding_dim
        --------------------- end final block

        output data with dimensions (batch, spatial_dims…, embedding_dim).
        """
        # settings
        norm_fn = layer.get_norm_layer(
            train=self.train, dtype=self.dtype, norm_type=self.norm_type
        )
        resblock_args = dict(
            norm_fn=norm_fn,
            conv_fn=self.conv_fn,
            dtype=self.dtype,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
        )

        # encoder
        # initial convolution
        x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)

        # main blocks = residual blocks + downsampling
        num_blocks = len(self.channel_multipliers)
        for i in range(num_blocks):
            filters = self.filters * self.channel_multipliers[i]
            for _ in range(self.num_res_blocks):
                x = ResBlock(filters, **resblock_args)(x)
            if i < num_blocks - 1:
                if self.conv_downsample:
                    # convolutional downsample
                    x = self.conv_fn(filters, kernel_size=(4, 4), strides=(2, 2))(x)
                else:
                    # average pooling
                    x = layer.dsample(x)
        
        # residual blocks + normalization + activation + 1x1 convolution to get the embedding dimension
        for _ in range(self.num_res_blocks):
            x = ResBlock(filters, **resblock_args)(x)
        x = norm_fn()(x)
        x = self.activation_fn(x)
        x = self.conv_fn(self.embedding_dim, kernel_size=(1, 1))(x)
        return x