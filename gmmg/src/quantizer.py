
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

from src import loss as loss_fn

class VectorQuantizer(nn.Module):
    """Basic vector quantizer."""

    config: ml_collections.ConfigDict
    train: bool
    dtype: int = jnp.float32

    @nn.compact
    def __call__(self, x, **kwargs):
        codebook_size = self.config.vqvae.codebook_size
        codebook = self.param(
            "codebook",
            jax.nn.initializers.variance_scaling(
                scale=1.0, mode="fan_in", distribution="uniform"
            ),
            (codebook_size, x.shape[-1]),
        )
        codebook = jnp.asarray(codebook, dtype=self.dtype)
        distances = jnp.reshape(
            loss_fn.squared_euclidean_distance(
                jnp.reshape(x, (-1, x.shape[-1])), codebook
            ),
            x.shape[:-1] + (codebook_size,),
        )
        encoding_indices = jnp.argmin(distances, axis=-1)
        encodings = jax.nn.one_hot(encoding_indices, codebook_size, dtype=self.dtype)
        quantized = self.quantize(encodings)
        result_dict = dict()
        if self.train:
            e_latent_loss = (
                jnp.mean((jax.lax.stop_gradient(quantized) - x) ** 2)
                * self.config.vqvae.commitment_cost
            )
            q_latent_loss = jnp.mean((quantized - jax.lax.stop_gradient(x)) ** 2)
            entropy_loss = 0.0
            if self.config.vqvae.entropy_loss_ratio != 0:
                entropy_loss = (
                    loss_fn.entropy_loss(
                        -distances,
                        loss_type=self.config.vqvae.entropy_loss_type,
                        temperature=self.config.vqvae.entropy_temperature,
                    )
                    * self.config.vqvae.entropy_loss_ratio
                )
            e_latent_loss = jnp.asarray(e_latent_loss, jnp.float32)
            q_latent_loss = jnp.asarray(q_latent_loss, jnp.float32)
            entropy_loss = jnp.asarray(entropy_loss, jnp.float32)
            loss = e_latent_loss + q_latent_loss + entropy_loss
            result_dict = dict(
                quantizer_loss=loss,
                e_latent_loss=e_latent_loss,
                q_latent_loss=q_latent_loss,
                entropy_loss=entropy_loss,
            )
            quantized = x + jax.lax.stop_gradient(quantized - x)

        result_dict.update(
            {
                "encodings": encodings,
                "encoding_indices": encoding_indices,
                "raw": x,
            }
        )
        return quantized, result_dict

    def quantize(self, z: jnp.ndarray) -> jnp.ndarray:
        codebook = jnp.asarray(self.variables["params"]["codebook"], dtype=self.dtype)
        return jnp.dot(z, codebook)

    def get_codebook(self) -> jnp.ndarray:
        return jnp.asarray(self.variables["params"]["codebook"], dtype=self.dtype)

    def decode_ids(self, ids: jnp.ndarray) -> jnp.ndarray:
        codebook = self.variables["params"]["codebook"]
        return jnp.take(codebook, ids, axis=0)