import jax
import jax.numpy as jnp

from collections import OrderedDict
import json


def pretty_params(params: OrderedDict, indent: int = 4) -> str:
    return json.dumps(
        jax.tree_map(lambda x: str(x.shape), params),
        indent = indent,
        sort_keys = False,
    )

def log_loss(predictions: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    lse = jax.scipy.special.logsumexp(predictions, axis=-1, keepdims=True)
    log_predictions = predictions - lse
    target_distribution = jax.nn.one_hot(labels, preds.shape[-1])
    return -jnp.mean(log_predictions * target_distribution)


def accuracy(predictions: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    return predictions.argmax(axis=-1) == labels
