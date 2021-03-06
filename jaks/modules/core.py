import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from collections import OrderedDict
import inspect
from typing import Optional, Tuple, Union


class Module:
    def modules(self):
        return

    def parameters(self, random_key: PRNGKey):
        return

    def init(
        self, 
        key: Union[int, PRNGKey], 
        params: Optional[OrderedDict] = None
        )-> Tuple[PRNGKey, OrderedDict]:
        
        if params is None:
            params = OrderedDict()

        if type(key) is int:
            key = jax.random.PRNGKey(key)
        key, subkey = jax.random.split(key)
        for name, param in self.parameters(subkey) or ():
            params[name] = param
        for name, module in self.modules() or ():
            key, params[name] = module.init(key, OrderedDict())
        return key, params

    def __call__(self, params: OrderedDict, *args, **kwargs):
        self._module_order = []
        for name, module in self.modules() or ():
            module.name = name
            setattr(self, name, module)
            self._module_order.append(name)
        if not hasattr(self, "name"):
            return self.forward(params, *args, **kwargs)
        return self.forward(params[self.name], *args, **kwargs)

    def forward(self, params: OrderedDict, x: jnp.ndarray) -> jnp.ndarray:
        for name in self._module_order:
            x = getattr(self, name)(params, x)
        return x

    def compile(self, batch: bool = True):
        @jax.jit
        def forward(params, *args, **kwargs):
            return self(params, *args, **kwargs)
        if batch:
            arg_count = len(inspect.signature(self.forward).parameters) - 1
            in_axes = [None] + [0] * arg_count
            return jax.vmap(forward, in_axes=in_axes)
        return forward
