# constants.py
import numpy as np
from numba import njit, prange

# Данные
T_293 = np.array([[1000, 1200, 1400, 1600, 1800, 2000],
                  [9.17440137,  9.423370201,  9.735918686,  10.01455683,  10.37994595,  10.5903492],
                  [29.35937721, 31.65155559,  30.2486405,   30.17815415,  29.94237192,  27.36357678],
                  [0.2326714,   0.2440502,    0.2528819,    0.2692589,	  0.2764174,    0.3013753],
                  [0.01385995,  0.01490037,   0.01370499,   0.01470922,   0.0150656,    0.02085652]])
T_310 = np.array([[1000, 1200, 1400, 1600, 1800],
                  [9.608579,    10.1564,	    10.48156,     10.75176,	    10.5243],
                  [19.53544,    20.56842378,  19.6038,      18.23266,	    22.76495312],
                  [0.1791479,   0.2147599,    0.1493049,    0.1797168,    0.2107221],
                  [0.025845326, 0.02878221,   0.05083024,   0.03752954,   None]])
T_323 = np.array([[1000, 1200, 1400, 1600, 1800],
                  [8.0366,      6.103743,     1.262356544,  3.568812289,  3.787543515],
                  [10.16,       10.49205139,  11.05093022,  10.84827952,  11.39313551],
                  [0.1512544,   0.1455957,    2.86294,      0.8830667,    0.1959008],
                  [0.1470505,   0.2757782,    0.1217176,    0.110712,     0.1518173]])
H_1000 = np.array([[293, 298, 302, 308, 313, 318, 323, 328, 333],
                   [9.139057,   9.351555,   9.691651,   9.98695,    10.44473,   8.067392,   3.632112,  2.565028,  3.181093],
                   [28.73127,   23.81016,   23.21427,   20.98695,   17.12141,   10.562,     10.74474,  11.40917,  11.06001],
                   [0.2326714,  0.1817891,  0.2213582,  0.1589079,  0.1216706,  0.06683201, 0.05,      0.3449964, 5],
                   [0.01385995, 0.01945772, 0.01818895, 0.01542324, 0.05003418, 0.1779732,  0.1140494, 0.1139074, 0.09414091]])

# Исходные параметры (Материал 1)
H_step = 10
H_lim = 4000
H_vals = np.arange(0, H_lim + 1, H_step)
T_vals_1 = np.linspace(290, 350, 601)
T_vals_2 = np.linspace(290, 350, 61)
T_init = 293

gamma = 1.76e7              # рад/(с·Oe)
alpha_1 = 1e-3
alpha_2 = 1.7e-2
h_IFE = 7500                # Ое
delta_t = 250e-15           # с

# Функции, зависящие от температуры (Материал 1)
@njit(cache=False)
def K_T(T):
    return 0.525 * (T - 370)**2

@njit(cache=False)
def chi_T(T):
    return 4.2e-7 * np.abs(T - 358)

# Загрузка данных для материала 1
m_array_1 = np.load('m_array_18.07.2025.npy')
M_array_1 = np.load('M_array_18.07.2025.npy')
chi_array_1 = chi_T(T_vals_1) if False else np.full_like(m_array_1, 8e-5) * 3.3
K_array_1 = K_T(T_vals_1)

# Альтернативные данные для Материала 2
m_array_2 = np.load('m_array_2.npy')
M_array_2 = np.load('M_array_2.npy')

# Для материала 2 зависимости K(T) и chi(T) заменяем константами
chi_const = 3.7e-4
K_const = 13500
chi_array_2 = np.full_like(m_array_2, chi_const)
K_array_2 = np.full_like(m_array_2, K_const)

def compute_frequencies(H_mesh, m_mesh, M_mesh, chi_mesh, K_mesh, gamma, alpha):
    abs_m = np.abs(m_mesh)

    w_H = gamma * H_mesh
    w_0_sq = gamma**2 * 2 * K_mesh / chi_mesh
    w_KK = gamma * abs_m / chi_mesh

    w_sq = w_0_sq + w_KK**2 / 4
    delta = w_H - w_KK / 2

    G = alpha * M_mesh * gamma / (2 * chi_mesh)
    W2 = w_sq - delta**2
    d_plus = delta - 1j * G
    d_minus = -delta - 1j * G

    w1 = np.sqrt(W2 + d_plus**2) + d_plus
    w2 = np.sqrt(W2 + d_minus**2) + d_minus
    w3 = -np.sqrt(W2 + d_plus**2) + d_plus
    w4 = -np.sqrt(W2 + d_minus**2) + d_minus

    roots = np.stack((w1, w2, w3, w4), axis=-1)
    sorted_indices = np.argsort(roots.real, axis=-1)[:, :, ::-1]
    sorted_roots = np.take_along_axis(roots, sorted_indices, axis=-1)

    f1, t1 = sorted_roots.real[:, :, 0] / (2 * np.pi * 1e9), -1e9 / sorted_roots.imag[:, :, 0]
    f2, t2 = sorted_roots.real[:, :, 1] / (2 * np.pi * 1e9), -1e9 / sorted_roots.imag[:, :, 1]
    return (f1, t1), (f2, t2)

# Вычисление частот
# --- FeFe ---
H_mesh_1, m_mesh_1 = np.meshgrid(H_vals, m_array_1)
_, M_mesh_1 = np.meshgrid(H_vals, M_array_1)
_, chi_mesh_1 = np.meshgrid(H_vals, chi_array_1)
_, K_mesh_1 = np.meshgrid(H_vals, K_array_1)

(f1_GHz, _), (f2_GHz, _) = compute_frequencies(
        H_mesh_1,
        m_mesh_1,
        M_mesh_1,
        chi_mesh_1,
        K_mesh_1,
        gamma,
        alpha_1)

# --- GdFe ---
H_mesh_2, m_mesh_2 = np.meshgrid(H_vals, m_array_2)
_, M_mesh_2 = np.meshgrid(H_vals, M_array_2)
_, chi_mesh_2 = np.meshgrid(H_vals, chi_array_2)
_, K_mesh_2 = np.meshgrid(H_vals, K_array_2)

def compute_phases(H_mesh, m_mesh, K_mesh, chi_mesh):
    abs_m = np.abs(m_mesh)
    m_cr = chi_mesh * H_mesh + (2 * K_mesh) / H_mesh
    theta_0 = np.where(H_mesh==0, np.nan, np.where(abs_m > m_cr, 0.0, np.arccos(abs_m / m_cr)))
      
    return theta_0


__all__ = [
    # сетки и оси
    'H_vals', 'T_vals_1', 'T_vals_2', 'T_init',
    # исходные одномерные массивы (нужны графикам)
    'm_array_1', 'M_array_1', 'm_array_2', 'M_array_2',
    'chi_array_1', 'K_array_1', 'chi_array_2', 'K_array_2',
    # физические константы
    'gamma', 'alpha_1', 'alpha_2', 'K_const', 'chi_const',
    'h_IFE', 'delta_t',
    # JIT-функция для частот
    'compute_frequencies', 'compute_phases',
    # частоты
    'f1_GHz', 'f2_GHz',
    # данные
    'T_293', 'T_310', 'T_323', 'H_1000',
]
