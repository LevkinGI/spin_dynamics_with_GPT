"""
Базовые константы и вспомогательные функции для модели частот/затуханий.

Файл содержит:
- экспериментальные точки (T_293, T_310, T_323, H_1000);
- температурные сетки и материалы (m_array, M_array, K_array);
- функции расчёта частот и времен затухания при заданных параметрах модели.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

# --- экспериментальные данные -------------------------------------------------
# Формат массивов:
#   [ось (H или T), частоты LF, частоты HF, tau_LF, tau_HF]
T_293 = np.array(
    [
        [1000, 1200, 1400, 1600, 1800, 2000],  # Э (Oe)
        [9.17440137, 9.423370201, 9.735918686, 10.01455683, 10.37994595, 10.5903492],
        [29.35937721, 31.65155559, 30.2486405, 30.17815415, 29.94237192, 27.36357678],
        [0.2326714, 0.2440502, 0.2528819, 0.2692589, 0.2764174, 0.3013753],
        [0.01385995, 0.01490037, 0.01370499, 0.01470922, 0.0150656, 0.02085652],
    ],
    dtype=float,
)

T_310 = np.array(
    [
        [1000, 1200, 1400, 1600, 1800],
        [9.608579, 10.1564, 10.48156, 10.75176, 10.5243],
        [19.53544, 20.56842378, 19.6038, 18.23266, 22.76495312],
        [0.1791479, 0.2147599, 0.1493049, 0.1797168, 0.2107221],
        [0.025845326, 0.02878221, 0.05083024, 0.03752954, np.nan],
    ],
    dtype=float,
)

T_323 = np.array(
    [
        [1000, 1200, 1400, 1600, 1800],
        [8.0366, 6.103743, 1.262356544, 3.568812289, 3.787543515],
        [10.16, 10.49205139, 11.05093022, 10.84827952, 11.39313551],
        [0.1512544, 0.1455957, 2.86294, 0.8830667, 0.1959008],
        [0.1470505, 0.2757782, 0.1217176, 0.110712, 0.1518173],
    ],
    dtype=float,
)

H_1000 = np.array(
    [
        [293, 298, 302, 308, 313, 318, 323, 328, 333],  # K
        [9.139057, 9.351555, 9.691651, 9.98695, 10.44473, 8.067392, 3.632112, 2.565028, 3.181093],
        [28.73127, 23.81016, 23.21427, 20.98695, 17.12141, 10.562, 10.74474, 11.40917, 11.06001],
        [0.2326714, 0.1817891, 0.2213582, 0.1589079, 0.1216706, 0.06683201, 0.05, 0.3449964, 5.0],
        [0.01385995, 0.01945772, 0.01818895, 0.01542324, 0.05003418, 0.1779732, 0.1140494, 0.1139074, 0.09414091],
    ],
    dtype=float,
)

# --- сетки и исходные материалы -----------------------------------------------
H_STEP = 10
H_LIM = 4000
H_VALS = np.arange(0, H_LIM + 1, H_STEP, dtype=float)
T_VALS = np.linspace(290, 350, 601, dtype=float)
T_INIT = 293.0

GAMMA = 1.76e7  # рад/(с·Oe)
ALPHA_DEFAULT = 1e-3

DATA_DIR = Path(__file__).parent


def k_T(temperature: Iterable[float] | float) -> np.ndarray:
    """Анизотропия как функция температуры."""
    T = np.asarray(temperature, dtype=float)
    return 0.522 * (T - 370.0) ** 2


def chi(m: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Вычисление магнитной восприимчивости."""
    m = np.asarray(m, dtype=float)
    M = np.asarray(M, dtype=float)
    denom = 1.0 - (m**2) / (M**2 + 1e-16)
    denom = np.where(denom == 0, np.nan, denom)
    return 1.0 / (12500.0 * denom)


# загружаем материалы из файлов
M_ARRAY = np.load(DATA_DIR / "M_array_18.07.2025.npy")
M_ARRAY = M_ARRAY.astype(float)
m_ARRAY = np.load(DATA_DIR / "m_array_18.07.2025.npy")
m_ARRAY = m_ARRAY.astype(float)
K_ARRAY = k_T(T_VALS)


def compute_frequencies(
    H_mesh: np.ndarray,
    m_mesh: np.ndarray,
    M_mesh: np.ndarray,
    K_mesh: np.ndarray,
    gamma: float = GAMMA,
    alpha: float = ALPHA_DEFAULT,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Расчёт частот/затуханий двух мод.

    Возвращает ((f1_GHz, tau1_ns), (f2_GHz, tau2_ns)).
    Частоты сортируются по убыванию; для получения LF/HF используйте
    min/max или сортировку по real-части.
    """
    abs_m = np.abs(m_mesh)
    chi_mesh = chi(m_mesh, M_mesh)

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

    f1 = sorted_roots.real[:, :, 0] / (2 * np.pi * 1e9)
    f2 = sorted_roots.real[:, :, 1] / (2 * np.pi * 1e9)
    tau1 = -1e9 / sorted_roots.imag[:, :, 0]
    tau2 = -1e9 / sorted_roots.imag[:, :, 1]
    return (f1, tau1), (f2, tau2)


def compute_phases(H_mesh: np.ndarray, m_mesh: np.ndarray, M_mesh: np.ndarray, K_mesh: np.ndarray) -> np.ndarray:
    """Угол theta_0 по исходной модели."""
    abs_m = np.abs(m_mesh)
    chi_mesh = chi(m_mesh, M_mesh)
    m_cr = chi_mesh * H_mesh + (2 * K_mesh) / H_mesh
    theta_0 = np.where(H_mesh == 0, np.nan, np.where(abs_m > m_cr, 0.0, np.arccos(abs_m / m_cr)))
    return theta_0


__all__ = [
    # сетки
    "H_STEP",
    "H_LIM",
    "H_VALS",
    "T_VALS",
    "T_INIT",
    # материалы
    "M_ARRAY",
    "m_ARRAY",
    "K_ARRAY",
    # константы
    "GAMMA",
    "ALPHA_DEFAULT",
    # функции модели
    "compute_frequencies",
    "compute_phases",
    "k_T",
    "chi",
    # эксперимент
    "T_293",
    "T_310",
    "T_323",
    "H_1000",
]
