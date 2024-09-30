import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


def skew(v: jnp.ndarray) -> jnp.ndarray:
    """Returns the skew symmetric matrix of a 3D vector."""
    return jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def vec(M: jnp.ndarray) -> jnp.ndarray:
    """Converts the matrix into a vector."""
    return M.flatten()


def devec(v: jnp.ndarray) -> jnp.ndarray:
    """Converts the vector into a 3x3 matrix."""
    return v.reshape(3, 3)


def rotation_matrix_from_vector(v):
    eps = 1e-5

    K = skew(v)

    theta = jnp.linalg.norm(jnp.abs(v) + eps)

    I = jnp.eye(3)
    K2 = K @ K
    sin_term = jnp.sin(theta) / theta
    cos_term = (1 - jnp.cos(theta)) / (theta**2)
    R = I + sin_term * K + cos_term * K2

    return R


def special_sign(v: jnp.ndarray) -> jnp.ndarray:
    """Returns the sign of the vector, with 0 mapped to 1."""
    return jnp.sign(v) + (v == 0)


def smooth_l1(x):
    """
    Smoothed l1 norm
    :param x: n-dim vector
    :return: scalar
    """
    delta = 1.0
    abs_errors = jnp.linalg.norm(x + 1e-6)
    quadratic = jnp.minimum(abs_errors, delta)
    linear = abs_errors - quadratic
    return 0.5 * quadratic**2 + delta * linear

def rot_from_quat(q: jnp.ndarray) -> Rotation:
    """
    Convert quaternion to rotation object (using another convention than
    scipy.spatial.transform.Rotation.from_quat)
    :param q: quaternion as (cos(theta/2), sin(theta/2) * axis)
    :return: rotation object
    """
    # permute the quaternion to match the convention of
    # scipy.spatial.transform.Rotation
    q = jnp.array([q[1], q[2], q[3], q[0]])
    return Rotation.from_quat(q)


def rot_to_quat(rot: Rotation) -> jnp.array:
    """
    Convert rotation object to quaternion
    :param rot: rotation object
    :return: rotation quaternion as (cos(theta/2), sin(theta/2) * axis)
    """
    q = rot.as_quat()
    return jnp.array([q[3], q[0], q[1], q[2]])


def normalize(a, a_min, a_max):
    """
    Maps input a from [a_min, a_max] to [-1, 1]
    """
    return 2 * (a - a_min) / (a_max - a_min) - 1