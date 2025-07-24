# simulator.py
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
from constants import gamma, h_IFE, delta_t

def _dynamics_factory(a: float, b: float, c: float, sign: int):
    @njit(cache=True, fastmath=True)
    def dynamics_const(t, y):
        theta, phi, dtheta, dphi = y
        ddtheta = -a * dtheta - b * theta - sign * c * dphi
        ddphi = -a * dphi - b * phi + sign * c * dtheta
        return (dtheta, dphi, ddtheta, ddphi)

    # ➜ вернём Python-обёртку — она вызывает уже скомпилированный код
    return dynamics_const.py_func

@njit(cache=True, fastmath=True)
def calc_coef(H: float, m: float, M: float, K: float,
              chi: float, alpha: float, kappa: float):
    a = alpha * M * gamma / chi
    b = (np.abs(m) * gamma**2 * H / chi - gamma**2 * H**2 + 2 * K * gamma**2 / chi)
    sign = 1 if m > 0 else -1
    c = 2 * gamma * H - sign * kappa * gamma**2 / chi

    return (a, b, c, sign)

def run_simulation(
        H: float,
        m: float,
        M: float,
        K: float,
        chi: float,
        alpha: float,
        kappa: float,
        simulation_time: float = 0.3e-9,
        num_points: int = 1001,
        method: str = 'DOP853',
        rtol: float = 1e-10,
        atol: float = 1e-12,
):
    # Начальные условия (в радианах и рад/с)
    theta_initial = 0.0
    phi_initial = 0.0
    dtheta_initial = 0.0
    dphi_initial = (gamma**2) * (H + abs(m) / chi) * h_IFE * delta_t

    a, b, c, sign = calc_coef(H, m, M, K, chi, alpha, kappa)
    dynamics = _dynamics_factory(a, b, c, sign)
                       
    y0 = [theta_initial, phi_initial, dtheta_initial, dphi_initial]
    t_eval = np.linspace(0, simulation_time, num_points)
    sol = solve_ivp(
        dynamics,
        (0.0, simulation_time),
        y0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y
