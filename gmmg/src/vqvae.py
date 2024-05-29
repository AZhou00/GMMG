from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn

import ml_collections

from src.encoder import Encoder
from src.decoder import Decoder
from src.quantizer import VectorQuantizer


class VQVAE(nn.Module):
    """VQVAE model."""

    config: ml_collections.ConfigDict
    train: bool
    dtype: int = jnp.float32
    activation_fn: Any = nn.relu

    def setup(self):
        """VQVAE setup."""
        self.quantizer = VectorQuantizer(
            config=self.config, train=self.train, dtype=self.dtype
        )
        self.encoder = Encoder(config=self.config, train=self.train, dtype=self.dtype)
        self.decoder = Decoder(
            config=self.config,
            train=self.train,
            output_dim=self.config.vqvae.output_dim,
            dtype=self.dtype,
        )

    def encode(self, input_dict):
        image = input_dict["image"]
        encoded_feature = self.encoder(image)
        quantized, result_dict = self.quantizer(encoded_feature)
        return quantized, result_dict

    def decode(self, x: jnp.ndarray) -> jnp.ndarray:
        reconstructed = self.decoder(x)
        return reconstructed

    def get_codebook_funct(self):
        return self.quantizer.get_codebook()

    def decode_from_indices(self, inputs):
        if isinstance(inputs, dict):
            ids = inputs["encoding_indices"]
        else:
            ids = inputs
        features = self.quantizer.decode_ids(ids)
        reconstructed_image = self.decode(features)
        return reconstructed_image

    def encode_to_indices(self, inputs):
        if isinstance(inputs, dict):
            image = inputs["image"]
        else:
            image = inputs
        encoded_feature = self.encoder(image)
        _, result_dict = self.quantizer(encoded_feature)
        ids = result_dict["encoding_indices"]
        return ids

    def __call__(self, input_dict):
        quantized, result_dict = self.encode(input_dict)
        outputs = self.decoder(quantized)
        return outputs, result_dict
