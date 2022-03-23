from dataclasses import dataclass
from typing import Iterable

from .core import Module
from .nonlin import RELU, ZScore, ScaledDotProductAttention
from .transform import Rotate, Translate, Scale


@dataclass
class Linear(Module):
    in_dims: int
    out_dims: int

    def modules(self):
        yield "weight", Rotate(self.in_dims, self.out_dims)
        yield "bias", Translate(self.out_dims)


@dataclass
class LayerNorm(Module):
    dims: int
    epsilon: float = 1e-05
    
    def modules(self):
        yield "zscore", ZScore(self.epsilon)
        yield "scale", Scale(self.dims)
        yield "translate", Translate(self.dims)


@dataclass
class MLP(Module):
    dims: Iterable[int]
    act_module: Module = RELU()
    
    def modules(self):
        for i in range(len(self.dims) - 1):
            if i > 0:
                yield "activation", self.act_module
            yield F"linear{i + 1}", Linear(self.dims[i], self.dims[i + 1])

@dataclass
class MultiHeadAttention(Module):
    dims: int
    heads: int
    head_dim_scale: float = 1.0

    def modules(self):
        head_dims = int(round(self.dims * self.head_dim_scale / self.heads))

        yield "qkv", Rotate(self.dims, 3 * self.heads * head_dims)
        yield "attention", ScaledDotProductAttention()
        yield "out", Rotate(self.heads * head_dims, self.dims)

    def forward(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        head_dims = int(round(self.dims * self.head_dim_scale / self.heads))

        qkv_fn = jax.vmap(self.qkv, in_axes=(None, 0))
        attn_fn = jax.vmap(self.attention, in_axes=(None, 0, 0, 0))
        attn_fn = jax.vmap(self.attention, in_axes=(None, 0, 0, 0))
        out_fn = jax.vmap(self.out, in_axes=(None, 0))

        qkv = qkv_fn(params, x)
        qkv = qkv.reshape(qkv.shape[0], 3, self.heads, head_dims)
        query, key, value = qkv[:,0], qkv[:,1], qkv[:,2]

        activations = attn_fn(params, query, key, value)
        activations = activations.reshape(qkv.shape[0], self.heads * head_dims)
        
        return out_fn(params, activations)
