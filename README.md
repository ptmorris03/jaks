# jaks

<p align="center">
    <img width="460" height="300" src="https://raw.githubusercontent.com/ptmorris03/jaks/main/jaks_logo.PNG">
</p>

```bash 
!pip install git+https://github.com/ptmorris03/jaks.git
```
```python3
import jax
import jax.numpy as jnp

import jaks
import jaks.modules as nn


#x: (dim,)
#weight: (out, dim)
#bias: (out,)
def linear(x, weight, bias):
    return jnp.matmul(weight, x) + bias

#x: (batch, c, dim1, dim2, ..., dimn)
#filters: (c_out, c_in, dim1, dim2, ..., dimn)
def convnd(x, filters, stride=1, padding='SAME'):
    if type(stride) is int:
        stride = (stride,) * (len(x.shape) - 2)
    if type(padding) is int:
        padding = ((padding, padding),) * (len(x.shape) - 2)
    return jax.lax.conv_general_dilated(x, filters, window_strides=stride, padding=padding)

#x: (dim1, dim2, dim..., dimn)
def zscore(x, axis=-1, eps=1e-5):
    centered = x - x.mean(axis=axis, keepdims=True)
    deviation = x.std(axis=axis, keepdims=True) + eps
    return centered / deviation

#x: (dim,)  
#weight: (dim,)
#bias: (dim,)
def layernorm(x, weight, bias):
    return zscore(x) * weight + bias

def random_init(random_key, shape):
    return jax.random.normal(random_key, shape) * sqrt(2 / shape[-1])


@dataclass
class MLP(nn.Module):
    dims: List[int]
    activation = jax.nn.relu

    def parameters(self, random_key):
        next_key = jaks.utils.PRNGSplitter(random_key)

        for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            yield F"weight{i}", random_init(next_key(), (d_out, d_in))
            yield F"bias{i}", jnp.zeros(d_out)

    def forward(self, params, x):
        for i in range(len(self.dims)):
            x = linear(x, params[F"weight{i}"], params[F"bias{i}"])
            if i < self.dims:
                x = self.activation(x)
        return x
        
        
mlp = MLP([784, 128, 1])
mlp_fn = mlp.compile(batch=True)
key, params = mlp_fn.init(key=4)
x = jnp.ones((32, 784))
y, grads = jax.value_and_grad(mlp_fn)(params, x)
```
