from typing import NamedTuple

import jax
import jax.numpy as jnp


class BodyDragParams(NamedTuple):
    horizontal_drag_coefficient: float
    vertical_drag_coefficient: float
    frontarea_x: float
    frontarea_y: float
    frontarea_z: float
    air_density: float


def compute_drag_force(state, key, params: BodyDragParams) -> jnp.ndarray:
    # unpack state
    v = state.v
    R = state.R

    # compute drag force
    v_body = R.T @ v
    rho = params.air_density
    area = jnp.array(
        [params.frontarea_x, params.frontarea_y, params.frontarea_z]
    )
    drag_coeff = jnp.array(
        [
            params.horizontal_drag_coefficient,
            params.horizontal_drag_coefficient,
            params.vertical_drag_coefficient,
        ]
    )

    # domain randomization
    drag_coeff = jax.random.uniform(
        key, drag_coeff.shape, minval=0.5 * drag_coeff, maxval=1.5 * drag_coeff
    )

    f_drag = -0.5 * rho * drag_coeff * area * v_body * jnp.abs(v_body)

    return f_drag
