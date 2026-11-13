import numpy as np

def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])

def quat_multiply(q, r):
    w, x, y, z = q
    a, b, c, d = r
    return np.array([w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c,
                     w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a])

def rotation(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def euler_to_quat(angles):
    r, p, y = np.asarray(angles)/2.
    cr, cp, cy = np.cos([r, p, y]); sr, sp, sy = np.sin([r, p, y])
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])

def quat_to_euler(q):
    R = rotation(q)
    return np.array([np.arctan2(R[2, 1], R[2, 2]), np.arcsin(np.clip(-R[2, 0], -1., 1.)),
                     np.arctan2(R[1, 0], R[0, 0])])

def coriolis(M, nu):

    momentum = M @ nu
    a, b = skew(momentum[:3]), skew(momentum[3:])
    return np.block([[np.zeros((3, 3)), -a], [-a, -b]])

class SixDOFDynamics:
    def __init__(self, config, current):
        self.cfg = p = config.physics
        self.current = current
        S = skew(p.cg_m)
        inertia_co = np.diag(p.inertia_cg_kg_m2) - p.mass_kg * S @ S
        self.M_RB = np.block([[p.mass_kg*np.eye(3), -p.mass_kg*S], [p.mass_kg*S, inertia_co]])
        self.M_A = np.diag(p.added_mass)
        self.M = self.M_RB + self.M_A
        np.linalg.cholesky(self.M)
        self.M_inv = np.linalg.inv(self.M)
        self.G = config.degradation

    def restoring(self, q):
        p = self.cfg
        R = rotation(q)
        gravity = R.T @ np.array([0., 0., p.mass_kg*p.gravity_m_s2])
        buoyancy = R.T @ np.array([0., 0., -p.water_density_kg_m3*p.gravity_m_s2*p.displaced_volume_m3])

        return -np.r_[gravity+buoyancy, np.cross(p.cg_m, gravity)+np.cross(p.cb_m, buoyancy)]

    def terms(self, state, t, action):
        position, q, nu = state[:3], state[3:7], state[7:]
        R = rotation(q)
        vc_n, grad, partial_t = self.current.sample(position, t)
        vc_b = R.T @ vc_n
        nu_c = np.r_[vc_b, np.zeros(3)]
        nu_r = nu - nu_c
        pos_dot = R @ nu[:3]

        nu_c_dot = np.r_[-np.cross(nu[3:], vc_b) + R.T @ (partial_t + grad @ pos_dot), np.zeros(3)]
        c_rb = coriolis(self.M_RB, nu) @ nu
        c_a = coriolis(self.M_A, nu_r) @ nu_r
        damping = (np.asarray(self.cfg.damping_linear) + np.asarray(self.cfg.damping_quadratic)*np.abs(nu_r))*nu_r
        tau_cmd = np.asarray(self.cfg.tau_max)*action
        return {"position_dot": pos_dot, "relative_velocity_body": nu_r, "current_ned_m_s": vc_n,
                "current_body_m_s": vc_b, "current_derivative_body": nu_c_dot,
                "coriolis_rigid": c_rb, "coriolis_added": c_a, "damping": damping,
                "restoring": self.restoring(q), "tau_commanded": tau_cmd, "tau_effective": self.G @ tau_cmd}

    def rhs(self, state, t, action):
        v = self.terms(state, t, action)
        acceleration = self.M_inv @ (v["tau_effective"] - v["coriolis_rigid"] - v["coriolis_added"]
                                    - v["damping"] - v["restoring"] + self.M_A @ v["current_derivative_body"])
        qdot = .5*quat_multiply(state[3:7], np.r_[0., state[10:13]])
        return np.r_[v["position_dot"], qdot, acceleration]

    def integrate(self, state, t, action, h):
        k1 = self.rhs(state, t, action)
        k2 = self.rhs(state + .5*h*k1, t + .5*h, action)
        k3 = self.rhs(state + .5*h*k2, t + .5*h, action)
        k4 = self.rhs(state + h*k3, t + h, action)
        result = state + h*(k1+2*k2+2*k3+k4)/6.
        qnorm = np.linalg.norm(result[3:7])
        if not np.all(np.isfinite(result)) or qnorm < 1e-10 or np.max(np.abs(result[7:])) > self.cfg.numerical_velocity_limit:
            raise FloatingPointError("6-DOF integration unstable; inspect physical parameters and reduce integration step")
        result[3:7] /= qnorm
        if np.dot(result[3:7], state[3:7]) < 0.:
            result[3:7] *= -1.
        return result
