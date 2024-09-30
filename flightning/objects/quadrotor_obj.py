import os
from functools import partial

import jax
import jax_dataclasses as jdc
import numpy as np
import yaml
from flightning.utils.pytrees import field_jnp, CustomPyTree
from jax import numpy as jnp
from flightning.simulation.model_body_drag import (
    BodyDragParams,
    compute_drag_force,
)
from flightning.utils.math import rotation_matrix_from_vector

from flightning.objects.quadrotor_simple_obj import (
    quadrotor_dyn as simple_dynamics,
)
from flightning.objects.quadrotor_simple_obj import (
    QuadrotorSimple,
)
import chex


@jdc.pytree_dataclass
class QuadrotorState(CustomPyTree):
    p: jax.Array = field_jnp([0.0, 0.0, 0.0])
    R: jax.Array = field_jnp(jnp.eye(3))
    v: jax.Array = field_jnp([0.0, 0.0, 0.0])
    omega: jax.Array = field_jnp([0.0, 0.0, 0.0])
    domega: jax.Array = field_jnp([0.0, 0.0, 0.0])
    motor_omega: jax.Array = field_jnp([0.0, 0.0, 0.0, 0.0])
    acc: jax.Array = field_jnp([0.0, 0.0, 0.0])
    dr_key: chex.PRNGKey = field_jnp(jax.random.key(0))

    def detached(self):
        return QuadrotorState(
            p=jax.lax.stop_gradient(self.p),
            R=jax.lax.stop_gradient(self.R),
            v=jax.lax.stop_gradient(self.v),
            omega=jax.lax.stop_gradient(self.omega),
            domega=jax.lax.stop_gradient(self.domega),
            motor_omega=jax.lax.stop_gradient(self.motor_omega),
            acc=jax.lax.stop_gradient(self.acc),
            dr_key=jax.lax.stop_gradient(self.dr_key),
        )

    def as_vector(self):
        return jnp.concatenate(
            [self.p, self.R.flatten(), self.v, self.omega, self.domega,
             self.motor_omega]
        )

    @classmethod
    def from_vector(cls, vector):
        p = vector[:3]
        R = vector[3:12].reshape(3, 3)
        v = vector[12:15]
        omega = vector[15:18]
        domega = vector[18:21]
        motor_omega = vector[21:]
        return cls(p, R, v, omega, domega, motor_omega)


class Quadrotor:
    """
    Full quadrotor model based on agilicious framework.
    Recommendation: Use from_yaml or from_name to create a quadrotor object.
    Note, the thrust map and drag augmentation is only valid for Kolibri

    >>> quad = Quadrotor.from_name("kolibri")
    >>> state = quad.default_state()
    >>> state_new = quad.step(state, 9.81 * quad._mass,
    >>>                       jnp.array([0.0, 0.0, 0.0]), 0.01)
    """

    def __init__(
            self,
            *,
            mass=0.75,  # [kg]
            tbm_fr=jnp.array([0.04, -0.04, 0.0]),  # [m]
            tbm_bl=jnp.array([-0.04,  0.04, 0.0]),  # [m]
            tbm_br=jnp.array([-0.04, -0.04, 0.0]),  # [m]
            tbm_fl=jnp.array([0.04,  0.04, 0.0]),  # [m]
            inertia=jnp.array([0.00014, 0.00016, 0.0002]),  # [kgm^2]
            motor_omega_min=150.0,  # [rad/s]
            motor_omega_max=4400.0,  # [rad/s]
            motor_tau=0.033,  # [s]
            motor_inertia=2.6e-7,  # [kgm^2]
            omega_max=jnp.array([6.0, 6.0, 4.0]),  # [rad/s]
            thrust_map=jnp.array([2e-7, 0.0,  0.0]),
            kappa=0.008,  # [Nm/N]
            thrust_min=0.0,  # [N]
            thrust_max=3.5,  # [N] per motor
            rotors_config="cross",
            dt_low_level=0.001,
    ):
        assert (
                rotors_config == "cross"
        ), "Only cross rotors configuration is supported"
        self._mass = mass
        self._tbm_fr = tbm_fr
        self._tbm_bl = tbm_bl
        self._tbm_br = tbm_br
        self._tbm_fl = tbm_fl
        self._inertia = inertia
        self._motor_omega_min = motor_omega_min
        self._motor_omega_max = motor_omega_max
        self._motor_tau = motor_tau
        self._motor_inertia = motor_inertia
        self._omega_max = omega_max
        self._thrust_map = thrust_map
        self._kappa = kappa
        self._thrust_min = thrust_min
        if thrust_min <= 0.0:
            self._thrust_min += thrust_map[0] * motor_omega_min ** 2
        self._thrust_max = thrust_max
        self._rotors_config = rotors_config
        self._dt_low_level = dt_low_level
        self._gravity = jnp.array([0, 0, -9.81])

        self.simple_model = QuadrotorSimple(mass=mass)

        # drag parameters
        self._drag_params = BodyDragParams(
            horizontal_drag_coefficient=1.04,
            vertical_drag_coefficient=1.04,
            frontarea_x=1.0e-3,
            frontarea_y=1.0e-3,
            frontarea_z=1.0e-2,
            air_density=1.2,
        )

    @classmethod
    def from_name(cls, name: str) -> "Quadrotor":
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "quadrotor_files/")

        if name == "example":
            filename += "example_quad.yaml"
        else:
            raise ValueError(f"Unknown quadrotor name: {name}")

        return cls.from_yaml(filename)

    @classmethod
    def from_yaml(cls, path: str) -> "Quadrotor":
        with open(path) as stream:
            try:
                config = yaml.safe_load(stream)
                return cls.from_dict(config)
            except yaml.YAMLError as exc:
                raise exc

    @classmethod
    def default_quadrotor(cls) -> "Quadrotor":
        return cls.from_name("example")

    @classmethod
    def from_dict(cls, config: dict) -> "Quadrotor":
        return cls(
            mass=config["mass"],
            tbm_fr=jnp.array(config["tbm_fr"]),
            tbm_bl=jnp.array(config["tbm_bl"]),
            tbm_br=jnp.array(config["tbm_br"]),
            tbm_fl=jnp.array(config["tbm_fl"]),
            inertia=jnp.array(config["inertia"]),
            motor_omega_min=config["motor_omega_min"],
            motor_omega_max=config["motor_omega_max"],
            motor_tau=config["motor_tau"],
            motor_inertia=config["motor_inertia"],
            omega_max=jnp.array(config["omega_max"]),
            thrust_map=jnp.array(config["thrust_map"]),
            kappa=config["kappa"],
            thrust_min=config["thrust_min"],
            thrust_max=config["thrust_max"],
            rotors_config=config["rotors_config"],
        )

    @property
    def hovering_motor_speed(self) -> float:
        return jnp.sqrt(self._mass * 9.81 / (4 * self._thrust_map[0]))

    def default_state(self):
        hovering_motor_speeds = jnp.ones(4) * self.hovering_motor_speed
        return QuadrotorState(motor_omega=hovering_motor_speeds)

    def create_state(self, p, R, v, **kwargs):
        hovering_motor_speed = jnp.ones(4) * self.hovering_motor_speed
        if "motor_omega" not in kwargs.keys():
            kwargs["motor_omega"] = hovering_motor_speed

        return QuadrotorState(p, R, v, **kwargs)

    @property
    def allocation_matrix(self):
        """
        maps [f1, f2, f3, f4] to [f_T, tau_x, tau_y, tau_z]
        """

        rotor_coordinates = np.stack(
            [self._tbm_fr, self._tbm_bl, self._tbm_br, self._tbm_fl]
        )
        x = rotor_coordinates[:, 0]
        y = rotor_coordinates[:, 1]

        return np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                y,
                -x,
                self._kappa * np.array([-1.0, -1.0, 1.0, 1.0]),
            ],
            dtype=np.float32,
        )

    def inertial_matrix(self):
        return np.diag(self._inertia)

    @property
    def allocation_matrix_inv(self):
        return np.linalg.inv(self.allocation_matrix)

    def step(
            self,
            state: QuadrotorState,
            f_d: jax.Array,
            omega_d: jax.Array,
            dt: jax.Array,
    ) -> QuadrotorState:
        """
        :param state: quadrotor state
        :param f_d: cumulative thrust [N]
        :param omega_d: commanded body rates [rad/s]
        :param dt: time step length [s]
        :return: next state of the quadrotor
        """

        @partial(jax.custom_jvp, nondiff_argnums=(3,))
        def _step(state, f_d, omega_d, dt):
            """Forward pass of the quadrotor dynamics."""

            # round dt to 5 decimal places to avoid numerical issues
            dt = np.round(dt, 5)
            if dt <= 0.0:
                return state

            def control_fn(state, _unused):
                """
                Low-level controller and dynamics.
                Runs by default at 1 kHz.
                """

                motor_omega_d = self._low_level_controller(
                    state, f_d, omega_d
                )

                state = self._dynamics(
                    state, motor_omega_d, self._dt_low_level
                )
                return state, None

            N = np.ceil(dt / self._dt_low_level).item()
            # check if dt is a multiple of dt_low_level
            assert np.isclose(
                N * self._dt_low_level, dt
            ), f"dt ({dt}) must be a multiple of dt_low_level ({self._dt_low_level})"

            state_new, _ = jax.lax.scan(control_fn, state, length=N)

            return state_new

        @_step.defjvp
        def _step_jvp(dt, primals, tangents):
            """Backward pass of the quadrotor dynamics."""

            state, f_d, omega_d = primals
            p, R, v = state.p, state.R, state.v

            state_dot, f_d_dot, omega_d_dot = tangents
            p_dot, R_dot, v_dot = state_dot.p, state_dot.R, state_dot.v

            state_new = _step(state, f_d, omega_d, dt)

            primals_simple = (p, R, v, f_d / self._mass, omega_d, dt)
            tangents_simple = (
                p_dot,
                R_dot,
                v_dot,
                f_d_dot / self._mass,
                omega_d_dot,
                0.0,
            )

            _, tan_out = jax.jvp(
                simple_dynamics, primals_simple, tangents_simple
            )

            p_tan, R_tan, v_tan = tan_out

            state_dot_new = state_dot.replace(
                p=p_tan, R=R_tan,
                v=v_tan, dr_key=state.dr_key
            )

            return state_new, state_dot_new

        return _step(state, f_d, omega_d, dt)

    def _dynamics(self, state: QuadrotorState, motor_omega_d, dt):
        # unpack state
        p = state.p
        R = state.R
        v = state.v
        omega = state.omega
        motor_omega = state.motor_omega

        # domain randomization keys
        key_thrust, key_drag = jax.random.split(state.dr_key)

        # position
        p_new = p + dt * v

        # orientation
        R_delta = rotation_matrix_from_vector(dt * omega)
        R_new = R @ R_delta

        # velocity
        # motor thrust
        thrust_map = self._thrust_map[0]
        thrust_map = jax.random.uniform(
            key_thrust,
            thrust_map.shape,
            minval=0.95 * thrust_map,
            maxval=1.05 * thrust_map,
        )

        f = thrust_map * motor_omega ** 2

        # Quadratic drag model
        f_drag = compute_drag_force(state, key_drag, self._drag_params)

        f_vec = jnp.array([0, 0, jnp.sum(f)]) + f_drag
        acc = self._gravity + R @ f_vec / self._mass
        v_new = v + dt * acc

        # angular acceleration
        dmotor_omega = 1 / self._motor_tau * (motor_omega_d - motor_omega)
        motor_directions = jnp.array([-1, -1, 1, 1])
        motor_inertia = self._motor_inertia
        inertia_torque = jnp.array(
            [0, 0, (dmotor_omega * motor_directions).sum() * motor_inertia]
        )

        # body torques and collective thrust
        J = self.inertial_matrix()
        J_inv = np.linalg.inv(J)
        f_T_and_tau = self.allocation_matrix @ f
        f_T, tau = f_T_and_tau[0], f_T_and_tau[1:]
        domega_new = J_inv @ (
                tau - jnp.cross(omega, J @ omega) + inertia_torque
        )
        omega_new = omega + dt * domega_new

        # motor dynamics
        motor_omega_new = (motor_omega - motor_omega_d) * jnp.exp(
            -dt / self._motor_tau
        ) + motor_omega_d
        # motor_omega_new = motor_omega_d

        motor_omega_new = jnp.clip(
            motor_omega_new, self._motor_omega_min, self._motor_omega_max
        )

        # noinspection PyArgumentList
        return state.replace(
            p=p_new,
            R=R_new,
            v=v_new,
            omega=omega_new,
            domega=domega_new,
            motor_omega=motor_omega_new,
            acc=acc,
        )

    def motor_omega_to_thrust(self, motor_omega):
        return self._thrust_map[0] * motor_omega ** 2

    def _low_level_controller(self, state, f_T, omega_cmd):
        # P control on body rates
        K = jnp.diag(jnp.array([20.0, 20.0, 41.0]))
        omega = state.omega
        omega_err = omega_cmd - omega
        body_torques_cmd = self.inertial_matrix() @ K @ omega_err + jnp.cross(
            omega, self.inertial_matrix() @ omega
        )
        alpha = jnp.concatenate([f_T[None], body_torques_cmd])
        f_cmd = self.allocation_matrix_inv @ alpha
        f_cmd = jnp.clip(f_cmd, self._thrust_min, self._thrust_max)
        motor_omega_d = jnp.sqrt(f_cmd / self._thrust_map[0])
        motor_omega_d = jnp.clip(
            motor_omega_d, self._motor_omega_min, self._motor_omega_max
        )
        return motor_omega_d

if __name__ == "__main__":

    quad = Quadrotor(mass=1.0)
    state = quad.default_state()
    f_d = jnp.array(1 * 9.81)
    omega_d = jnp.array([0.0, 0.0, 0.01])
    dt = 1.0
    state_new = quad.step(state, f_d, omega_d, dt)
    print(state_new.p)

