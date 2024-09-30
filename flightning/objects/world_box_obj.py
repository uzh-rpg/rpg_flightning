import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from flightning.utils.pytrees import field_jnp


@jdc.pytree_dataclass
class WorldBox:
    """
    Represents the world box.

    >>> world_box = WorldBox(min=jnp.array([-10.0, -10.0, -10.0]),
    >>>                      max=jnp.array([10.0, 10.0, 10.0]))
    >>> position = jnp.array([0.0, 0.0, 0.0])
    >>> is_inside = world_box.contains(position)
    """

    min: jnp.array = field_jnp([-10.0, -10.0, -10.0])
    max: jnp.array = field_jnp([10.0, 10.0, 10.0])

    def contains(self, x: jnp.array) -> jax.Array:
        is_inside = jnp.logical_and(
            jnp.all(x >= self.min), jnp.all(x <= self.max)
        )
        return is_inside

    @classmethod
    def from_yaml_entry(cls, entry: list) -> "WorldBox":
        # noinspection PyArgumentList
        return cls(
            min=jnp.array(entry[:3]),
            max=jnp.array(entry[3:]),
        )