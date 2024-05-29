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

class Decoder(nn.Module):
    """Decoder Blocks."""

    config: ml_collections.ConfigDict
    train: bool
    output_dim: int = 1
    dtype: Any = jnp.float32

    def setup(self):
        self.filters = self.config.vqvae.filters
        self.num_res_blocks = self.config.vqvae.num_res_blocks
        self.channel_multipliers = self.config.vqvae.channel_multipliers
    
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

        # decoder
        filters = self.filters * self.channel_multipliers[-1]
        x = self.conv_fn(filters, kernel_size=(3, 3), use_bias=True)(x)
        for _ in range(self.num_res_blocks):
            x = ResBlock(filters, **resblock_args)(x)

        num_blocks = len(self.channel_multipliers)
        for i in reversed(range(num_blocks)):
            filters = self.filters * self.channel_multipliers[i]
            for _ in range(self.num_res_blocks):
                x = ResBlock(filters, **resblock_args)(x)
            if i > 0:
                x = layer.upsample(x, 2)
                x = self.conv_fn(filters, kernel_size=(3, 3))(x)
        x = norm_fn()(x)
        x = self.activation_fn(x)
        x = self.conv_fn(self.output_dim, kernel_size=(3, 3))(x)
        return x