from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .core import Module


class RELU(Module):
    def forward(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.maximum(x, 0)


class QuickGELU(Module):
    def forward(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.nn.sigmoid(1.702 * x)


@dataclass
class ZScore(Module):
    epsilon: float = 1e-05

    def forward(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + self.epsilon
        return (x - mean) / std


class ScaledDotProductAttention(Module):
    def forward(
        self, 
        params: Any, 
        query: jnp.ndarray, 
        key: jnp.ndarray, 
        value: jnp.ndarray
        ) -> jnp.ndarray:

        scale = 1 / jnp.sqrt(key.shape[-1])
        dot_prod = jnp.matmul(query, key.T)
        attn = jax.nn.softmax(jnp.multiply(dot_prod, scale))
        return jnp.matmul(attn, value)
