from typing import List, Any, Callable, NewType, TypeVar

import chex
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from flax.struct import field
from jax.scipy.spatial.transform import Rotation
from typing import Union
import numpy as np

T = TypeVar("T")
TCustomPyTree = TypeVar("TCustomPyTree", bound="CustomPyTree")


@jdc.pytree_dataclass
class CustomPyTree:
    def replace(self: TCustomPyTree, **kwargs) -> TCustomPyTree:
        return jdc.replace(self, **kwargs)


def tree_select(predicate, true_tree: T, false_tree: T) -> T:
    return jax.tree.map(
        lambda leave_a, leave_b: jax.lax.select(predicate, leave_a, leave_b),
        true_tree,
        false_tree,
    )


def field_jnp(container):
    """
    Uitility to create dataclass field for jnp.array
    :param container: list or tuple or array
    :return: filed with default_factory

    Example:

    >>> from flax import struct
    >>> import jax.numpy as jnp
    >>> from flightning.utils.pytrees import field_jnp
    >>> class Test(struct.PyTreeNode):
        ...     a: jnp.ndarray = field_jnp([1, 2, 3])
    >>> Test()
    """
    return field(default_factory=lambda: jnp.array(container))


def field_np(container):

    return field(default_factory=lambda: np.array(container))


def stack_pytrees(pytrees: List[Any]) -> Any:
    """
    Stack a list of PyTrees into a single PyTree.

    Example:

    >>> import jax
    >>> from jax import vmap
    >>> import jax.numpy as jnp
    >>> from flax.struct import PyTreeNode
    >>> import jax.tree as tree
    >>>
    >>>
    >>> class State(PyTreeNode):
    >>>     reward: jnp.ndarray
    >>>
    >>> @vmap
    >>> def test(state: State):
    >>>     return state.reward + 3
    >>>
    >>> state_list = [State(1), State(2)]
    >>> new_pytree = stack_pytrees(state_list)
    >>> res = test(new_pytree)
    """
    return jax.tree.map(lambda *args: jnp.stack(args), *pytrees)


def pytree_get_item(pytree: T, idx: Union[int, tuple, None]) -> T:
    return jax.tree.map(lambda leave: leave[idx], pytree)


def pytree_roll(pytree: T, shift: int, axis: int) -> T:
    """
    Roll the pytree along the axis axis
    :param pytree: pytree
    :param shift: shift amount
    :return: rolled pytree
    """
    return jax.tree.map(lambda x: jnp.roll(x, shift, axis), pytree)


def pytree_at_set(pytree: T, idx: Union[int, tuple], value: Any) -> T:
    """
    Set value at index in pytree
    :param pytree: pytree
    :param idx: index
    :param value: value
    :return: pytree with value set at index
    """
    return jax.tree.map(lambda x, y: x.at[idx].set(y), pytree, value)
