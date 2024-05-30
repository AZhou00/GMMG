
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

        # drawing numbers from a uniform interval 
        # with variance = scale/codebook_size, 
        # or std = sqrt(variance scale/codebook_size)
        # this call on self.param also initializes the `codebook` in the `param` dict
        # 
        # Parameter Storage:
        #   When self.param is called, Flax checks if a parameter with the given name ("codebook") already exists in the model's parameter dictionary. 
        #   If it doesn't exist, Flax initializes the parameter using the provided initializer and shape, and then stores it in the parameter dictionary.
        #
        # codebook: (codebook_size, embedding_dim)
        # i.e., each codebook entry lives in the embedding space,
        # representing a vector in the embedding space
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

        # shape: (batch, spatial_dims…)
        encoding_indices = jnp.argmin(distances, axis=-1)
        # shape: (batch, spatial_dims…, codebook_size)
        # for each batch and spatial_dims combinition, 
        # there is a vector of length codebook_size with 1 at the index of the closest codebook entry
        # and 0 elsewhere
        encodings = jax.nn.one_hot(encoding_indices, codebook_size, dtype=self.dtype)
        # jnp.dot(encodings, codebook), 
        # (batch, spatial_dims…, codebook_size) @ (codebook_size, embedding_dim) = (batch, spatial_dims…, embedding_dim)
        quantized = self.quantize(encodings)
        result_dict = dict()

        # if training, calculate loss
        if self.train:
            # e_latent_loss: commitment loss, this trains the encoder to move the latents to the closest codebook entry
            e_latent_loss = (
                jnp.mean((jax.lax.stop_gradient(quantized) - x) ** 2)
                * self.config.vqvae.commitment_cost
            )

            # q_latent_loss: this trains the codebook
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

            # bridging the gradient flow
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