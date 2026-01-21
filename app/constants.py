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

# --- сетки и исходные материалы -----------------------------------------------
H_STEP = 10
H_LIM = 4000
H_VALS = np.arange(0, H_LIM + 1, H_STEP, dtype=float)
T_VALS = np.linspace(290, 350, 601, dtype=float)
T_INIT = 293.0

GAMMA = 1.76e7  # рад/(с·Oe)
ALPHA_DEFAULT = 1e-3
LAMBDA_WEISS = 12500.0

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR


def k_T(temperature: Iterable[float] | float) -> np.ndarray:
    """Анизотропия как функция температуры."""
    T = np.asarray(temperature, dtype=float)
    return 0.522 * (T - 370.0) ** 2


def chi(m: np.ndarray, M: np.ndarray, *, lambda_weiss: float = LAMBDA_WEISS) -> np.ndarray:
    """Вычисление магнитной восприимчивости."""
    m = np.asarray(m, dtype=float)
    M = np.asarray(M, dtype=float)
    denom = 1.0 - (m**2) / (M**2 + 1e-16)
    denom = np.where(denom == 0, np.nan, denom)
    return 1.0 / (lambda_weiss * denom)


# загружаем материалы из файлов
SUM_MAG_ARRAY = np.load(DATA_DIR / "sum_magnetizations.npy")
SUM_MAG_ARRAY = SUM_MAG_ARRAY.astype(float)
DIFF_MAG_ARRAY = np.load(DATA_DIR / "diff_magnetizations.npy")
DIFF_MAG_ARRAY = DIFF_MAG_ARRAY.astype(float)
K_ARRAY = k_T(T_VALS)


def compute_frequencies(
    H_mesh: np.ndarray,
    m_mesh: np.ndarray,
    M_mesh: np.ndarray,
    K_mesh: np.ndarray,
    gamma: float = GAMMA,
    alpha: float = ALPHA_DEFAULT,
    lambda_weiss: float = LAMBDA_WEISS,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Расчёт частот/затуханий двух мод.

    Возвращает ((f1_GHz, tau1_ns), (f2_GHz, tau2_ns)).
    Частоты сортируются по убыванию; для получения LF/HF используйте
    min/max или сортировку по real-части.
    """
    abs_m = np.abs(m_mesh)
    chi_mesh = chi(m_mesh, M_mesh, lambda_weiss=lambda_weiss)

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


def compute_phases(
    H_mesh: np.ndarray,
    m_mesh: np.ndarray,
    M_mesh: np.ndarray,
    K_mesh: np.ndarray,
    *,
    lambda_weiss: float = LAMBDA_WEISS,
) -> np.ndarray:
    """Угол theta_0 по исходной модели."""
    abs_m = np.abs(m_mesh)
    chi_mesh = chi(m_mesh, M_mesh, lambda_weiss=lambda_weiss)
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
    "SUM_MAG_ARRAY",
    "DIFF_MAG_ARRAY",
    "K_ARRAY",
    # константы
    "GAMMA",
    "ALPHA_DEFAULT",
    "LAMBDA_WEISS",
    # функции модели
    "compute_frequencies",
    "compute_phases",
    "k_T",
    "chi",
]
