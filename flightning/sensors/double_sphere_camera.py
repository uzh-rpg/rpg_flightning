from enum import Enum
from typing import Union

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import yaml
from jax.scipy.spatial.transform import Rotation

from flightning import FLIGHTNING_PATH
from flightning.utils.pytrees import field_jnp, CustomPyTree


class CameraNames(Enum):
    EXAMPLE = "example"


EXAMPLE = FLIGHTNING_PATH + "/sensors/camera_files/example_cam.yaml"


@jdc.pytree_dataclass
class CameraState(CustomPyTree):
    # world position in camera frame
    p_CW: jnp.ndarray = field_jnp([0.0, 0.0, 0.0])
    # rotation from world to camera frame
    R_CW: jnp.ndarray = field_jnp(jnp.eye(3))


class DoubleSphereCamera:

    def __init__(
            self,
            fx: float,
            fy: float,
            cx: float,
            cy: float,
            alpha: float,
            xi: float,
            width: int,
            height: int,
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.alpha = alpha
        self.xi = xi
        self.width = width
        self.height = height
        self.pitch = 0.

    @property
    def pitch(self):
        """Pitch angle in degrees"""
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        self._pitch = value
        # camera points in x direction of quad frame
        rot_CprimeB = Rotation.from_euler(
            "XYZ", jnp.array([90, 0, 90]), degrees=True
        )
        rot_cam = Rotation.from_euler("Y", jnp.array(self._pitch), degrees=True)
        self.rot_CB = rot_CprimeB * rot_cam

    @classmethod
    def from_name(cls, name: Union[str, CameraNames]) -> "DoubleSphereCamera":

        if isinstance(name, CameraNames):
            name = name.value

        if name == "example":
            return cls.from_yaml(EXAMPLE)
        else:
            raise ValueError(f"Unknown camera name: {name}")

    @classmethod
    def from_yaml(cls, path: str) -> "DoubleSphereCamera":
        with open(path) as stream:
            try:
                config = yaml.safe_load(stream)
                return cls.from_dict(config)
            except yaml.YAMLError as exc:
                raise exc

    @classmethod
    def from_dict(cls, config: dict) -> "DoubleSphereCamera":
        return cls(
            xi=config["intrinsics"][0],
            alpha=config["intrinsics"][1],
            fx=config["intrinsics"][2],
            fy=config["intrinsics"][3],
            cx=config["intrinsics"][4],
            cy=config["intrinsics"][5],
            width=config["resolution"][0],
            height=config["resolution"][1],
        )

    def project_points(
            self, points: jax.Array, camera_state: CameraState
    ) -> jax.Array:
        """
        Project points from world frame to image frame
        :param points: Nx3 array of points in world frame
        :param camera_state: CameraState object
        :return: Nx3 array of projected points in image frame + validity flag
        """

        # Transform points to camera coordinates
        rot_CW = Rotation.from_matrix(camera_state.R_CW)
        points_C = rot_CW.apply(points) + camera_state.p_CW

        # Calculate norms and intermediate points
        d1 = jnp.linalg.norm(points_C, axis=1)
        points_C_zxi = points_C.at[:, 2].add(d1 * self.xi)
        d2 = jnp.linalg.norm(points_C_zxi, axis=1)

        # Calculate divisor for projection
        div = self.alpha * d2 + (1 - self.alpha) * points_C_zxi[:, 2]

        # Compute projected points
        u = self.fx * (points_C[:, 0] / div) + self.cx
        v = self.fy * (points_C[:, 1] / div) + self.cy

        # Check validity of points
        w1 = jax.lax.select(
            self.alpha <= 0.5,
            self.alpha / (1 - self.alpha),
            (1 - self.alpha) / self.alpha,
        )
        w2 = (w1 + self.xi) / jnp.sqrt(2 * w1 * self.xi + self.xi ** 2 + 1)
        predicates = jnp.array(
            [
                # check if point is in front of the camera
                points_C[:, 2] > 0,
                # check if projection is valid
                points_C[:, 2] > -w2 * d1,
                # check if the point is in the image
                u >= 0,
                u < self.width,
                v >= 0,
                v < self.height,
            ]
        )
        valid = jnp.all(predicates, axis=0)

        # Combine projected points and validity flag
        projected_points = jnp.column_stack((u, v, valid.astype(float)))

        return projected_points

    def update_pose(
            self, state: CameraState, p_WB: jax.Array, R_WB: jax.Array
    ) -> CameraState:
        rot_WB = Rotation.from_matrix(R_WB)
        rot_BW = rot_WB.inv()
        rot_CW = self.rot_CB * rot_BW
        p_CW = rot_CW.apply(-p_WB)

        state_new = state.replace(p_CW=p_CW, R_CW=rot_CW.as_matrix())

        return state_new

    def project_points_with_pose(
            self, points: jax.Array, p_WB: jax.Array, R_WB: jax.Array
    ) -> jax.Array:
        # noinspection PyArgumentList
        state = CameraState(p_CW=jnp.zeros(3), R_CW=jnp.eye(3))
        state_new = self.update_pose(state, p_WB, R_WB)
        return self.project_points(points, state_new)


if __name__ == "__main__":
    camera = DoubleSphereCamera.from_name(CameraNames.EXAMPLE)
    points = jnp.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 1000.0],
            [100.0, 8.0, -1.0],
        ]
    )
    state = CameraState()
    projected_points = camera.project_points(points, state)
    print(projected_points)
    state_new = camera.update_pose(
        state, jnp.array([0.0, 0.0, 0.0]), jnp.eye(3)
    )
    print(state_new)
