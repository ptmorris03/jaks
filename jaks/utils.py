import jax
import jax.numpy as jnp
from jaxlib.xla_extension import DeviceArray

from collections import OrderedDict
import json
from typing import Union


class PRNGSplitter:
    def __init__(self, seed: Union[int, DeviceArray]):
        self.key = seed

        if type(self.key) is int:
            self.key = jax.random.PRNGKey(self.key)

    def __call__(self):
        self.key, key2 = jax.random.split(self.key)
        return key2


def pretty_params(params: OrderedDict, indent: int = 4) -> str:
    shape_map = jax.tree_map(lambda x: str(x.shape), params)
    return json.dumps(shape_map, indent=indent, sort_keys=False)

def log_loss(predictions: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    lse = jax.scipy.special.logsumexp(predictions, axis=-1, keepdims=True)
    log_predictions = predictions - lse
    target_distribution = jax.nn.one_hot(labels, predictions.shape[-1])
    return -jnp.mean(log_predictions * target_distribution)

def accuracy(predictions: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    return predictions.argmax(axis=-1) == labels

def sgd(params: OrderedDict, grads: OrderedDict, lr: float) -> OrderedDict:
    def map_fn(param, grad):
        return param - lr * grad
    return jax.tree_multimap(map_fn, params, grads)
