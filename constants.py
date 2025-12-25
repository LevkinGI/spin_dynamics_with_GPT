# constants.py
import numpy as np

# Данные. Строки: поле/температура, нч в ГГц, вч, затухание нч в нс, затухание вч.
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

# Исходные параметры
H_step = 10
H_lim = 4000
H_vals = np.arange(0, H_lim + 1, H_step)
T_vals = np.linspace(290, 350, 601)
T_init = 293

gamma = 1.76e7              # рад/(с·Oe)
alpha = 1e-3
h_IFE = 7500                # Ое
delta_t = 250e-15           # с

def _K_T(T):
    return 0.522 * (T - 370)**2
  
def _chi(m, M):
    return 1 / (12500 * (1 - m**2 / M**2))

# Загрузка данных
m_array = np.load('m_array_18.07.2025.npy')
M_array = np.load('M_array_18.07.2025.npy')
K_array = K_T(T_vals)

def compute_frequencies(H_mesh, m_mesh, M_mesh, K_mesh, gamma, alpha):
    abs_m = np.abs(m_mesh)
    chi_mesh = _chi(m_mesh, M_mesh)

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
H_mesh, m_mesh = np.meshgrid(H_vals, m_array)
_, M_mesh = np.meshgrid(H_vals, M_array)
_, chi_mesh = np.meshgrid(H_vals, chi_array)
_, K_mesh = np.meshgrid(H_vals, K_array)

(f1_GHz, _), (f2_GHz, _) = compute_frequencies(
        H_mesh,
        m_mesh,
        M_mesh,
        chi_mesh,
        K_mesh,
        gamma,
        alpha)

def compute_phases(H_mesh, m_mesh, K_mesh, chi_mesh):
    abs_m = np.abs(m_mesh)
    m_cr = chi_mesh * H_mesh + (2 * K_mesh) / H_mesh
    theta_0 = np.where(H_mesh==0, np.nan, np.where(abs_m > m_cr, 0.0, np.arccos(abs_m / m_cr)))
      
    return theta_0


__all__ = [
    # сетки и оси
    'H_vals', 'T_vals',, 'T_init',
    # исходные одномерные массивы (нужны графикам)
    'm_array', 'M_array', 'K_array',
    # физические константы
    'gamma', 'alpha', 'h_IFE', 'delta_t',
    # функции
    'compute_frequencies', 'compute_phases',
    # частоты
    'f1_GHz', 'f2_GHz',
    # данные
    'T_293', 'T_310', 'T_323', 'H_1000',
]
