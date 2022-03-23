from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Union

import jax
import jax.numpy as jnp

from .core import Module


@dataclass
class Residual(Module):
    module: Union[Module, Iterable[Module]]

    def modules(self):
        try:
            for i, m in enumerate(self.module):
                yield F"residual_module{i+1}", m
        except TypeError:
            yield "residual_module1", self.module

    def forward(self, params: OrderedDict, x: jnp.ndarray) -> jnp.ndarray:
        try:
            y = x
            for i, m in enumerate(self.module):
                y = getattr(self, F"residual_module{i+1}")(params, y)
            return y + x
        except TypeError:
            return self.residual_module1(params, x) + x


@dataclass
class ResNetStack(Module):
    module: Module
    depth: int

    def modules(self):
        residual_layer = Residual(self.module)
        for layer in range(1, self.depth + 1):
            yield F"layer{layer}", residual_layer


@dataclass
class Loop(Module):
    module: Module
    iterations: int

    def modules(self):
        yield "loop_module", self.module

    def forward(self, params: OrderedDict, x: jnp.ndarray) -> jnp.ndarray:
        def step_fn(i, x):
            return self.loop_module(params, x)
        return jax.lax.fori_loop(0, self.iterations, step_fn, x)
