from collections import OrderedDict
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from .core import Module


@dataclass
class Translate(Module):
    dims: int

    def parameters(self, random_key: PRNGKey):
        yield "vector", jnp.zeros(self.dims)

    def forward(self, params: OrderedDict, x: jnp.ndarray) -> jnp.ndarray:
        return x + params["vector"]


@dataclass
class Scale(Module):
    dims: int

    def parameters(self, random_key: PRNGKey):
        yield "vector", jnp.ones(self.dims)

    def forward(self, params: OrderedDict, x: jnp.ndarray) -> jnp.ndarray:
        return x * params["vector"]


@dataclass
class DotProduct(Module):
    dims: int
    init_scale: float = 1e-2

    def parameters(self, random_key: PRNGKey):
        vector = jax.random.normal(random_key, (self.dims,))
        yield "vector", vector * self.init_scale

    def forward(self, params: OrderedDict, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(x, params["vector"])


@dataclass
class Rotate(Module):
    in_dims: int
    out_dims: int
    init_scale: float = 1e-2

    def parameters(self, random_key: PRNGKey):
        matrix = jax.random.normal(random_key, (self.out_dims, self.in_dims))
        yield "matrix", matrix * self.init_scale

    def forward(self, params: OrderedDict, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(params["matrix"], x)
