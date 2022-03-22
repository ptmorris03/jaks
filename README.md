# jaks
```bash 
pip install git+https://github.com/ptmorris03/jaks.git
```
```python3
from dataclasses import dataclass
from typing import Iterable

import jax
import jax.numpy as jnp

import jaks
import jaks.modules as nn


@dataclass
class Linear(nn.Module):
    in_dims: int
    out_dims: int

    def modules(self):
        yield "weight", nn.Rotate(self.in_dims, self.out_dims)
        yield "bias", nn.Translate(self.out_dims)


@dataclass
class MLP(nn.Module):
    dims: Iterable[int]
    act_module: Module = nn.RELU()
    
    def modules(self):
        for i in range(len(self.dims) - 1):
            if i > 0:
                yield "activation", self.act_module
            yield F"linear{i + 1}", nn.Linear(self.dims[i], self.dims[i + 1])
            

@dataclass
class ResidualGate(nn.Module):
    module: nn.Module
    gate_input: bool = False
        
    def modules(self):
        yield "module", self.module
        
    def parameters(self, key):
        yield "gate", jnp.ones(1)
        
    def forward(self, params, x):
        output = self.module(params, x)
        gated_output = params["gate"] * output
        if self.gate_input:
            x = (1 - params["gate"]) * x
        return gated_output + x
        
        
mlp = MLP([784, 128, 10])
mlp_fn = mlp.compile(batch=True)
key, params = mlp_fn.init(key=42)
x = jnp.ones((32, 784))
y, grads = jax.value_and_grad(mlp_fn)(params, x)
```
