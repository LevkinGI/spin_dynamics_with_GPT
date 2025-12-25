"""
Подбор коэффициентов k_M, k_m, k_K и alpha по экспериментальным точкам.

Скрипт строит вектор невязок для всех доступных наборов данных и запускает
`scipy.optimize.least_squares`. После подбора сохраняется сводный график
(`summary_plot.html`), который строится функцией build_summary_figure().
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .constants import (
    ALPHA_DEFAULT,
    GAMMA,
    K_ARRAY,
    M_ARRAY,
    T_VALS,
    compute_frequencies,
    m_ARRAY,
)
from .plotting import SeriesData, build_summary_figure

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "approximation.log"
DATA_DIR = BASE_DIR / "data"
TAU_WEIGHT_FALLBACK = 0.2
logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Настройка логирования в файл и консоль (append)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if logger.handlers:
        return  # уже настроено

    formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


@dataclass
class ModelParameters:
    k_M: float = 1.0
    k_m: float = 1.0
    k_K: float = 1.0
    alpha: float = ALPHA_DEFAULT

    def as_array(self) -> np.ndarray:
        return np.array([self.k_M, self.k_m, self.k_K, self.alpha], dtype=float)

    @classmethod
    def from_array(cls, arr: Sequence[float]) -> "ModelParameters":
        return cls(k_M=float(arr[0]), k_m=float(arr[1]), k_K=float(arr[2]), alpha=float(arr[3]))


@dataclass
class Observation:
    H: float
    T: float
    f_lf: float | None
    f_hf: float | None
    tau_lf: float | None
    tau_hf: float | None
    err_f_lf: float | None = None
    err_f_hf: float | None = None
    err_tau_lf: float | None = None
    err_tau_hf: float | None = None


@dataclass
class ParsedSeries:
    """Экспериментальные данные из одного Excel-файла."""

    name: str
    axis_label: str
    axis_values: np.ndarray
    observations: List[Observation]
    exp_f_lf: np.ndarray
    exp_f_hf: np.ndarray
    exp_tau_lf: np.ndarray
    exp_tau_hf: np.ndarray


def _split_modes(f1: float, tau1: float, f2: float, tau2: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Возвращает ((f_LF, tau_LF), (f_HF, tau_HF))."""
    modes = sorted(((f1, tau1), (f2, tau2)), key=lambda x: x[0])
    return modes[0], modes[1]


def _predict_single_point(H: float, T: float, params: ModelParameters) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Расчёт частот и времен затухания для одной точки (H, T)."""
    m_val, M_val, K_val = _materials_at_temperature(T)

    m_scaled = params.k_m * m_val
    M_scaled = params.k_M * M_val
    K_scaled = params.k_K * K_val

    (f1, tau1), (f2, tau2) = compute_frequencies(
        H_mesh=np.array([[H]], dtype=float),
        m_mesh=np.array([[m_scaled]], dtype=float),
        M_mesh=np.array([[M_scaled]], dtype=float),
        K_mesh=np.array([[K_scaled]], dtype=float),
        gamma=GAMMA,
        alpha=params.alpha,
    )
    (f_lf, tau_lf), (f_hf, tau_hf) = _split_modes(float(f1[0, 0]), float(tau1[0, 0]), float(f2[0, 0]), float(tau2[0, 0]))
    return (f_lf, tau_lf), (f_hf, tau_hf)


def _build_observations_and_series(data_dir: Path) -> Tuple[List[Observation], List[ParsedSeries]]:
    series_list: list[ParsedSeries] = []
    observations: list[Observation] = []

    files = sorted(list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls")))
    if not files:
        logger.error("В каталоге %s не найдено Excel-файлов (*.xlsx / *.xls).", data_dir)
        raise FileNotFoundError(f"В каталоге {data_dir} не найдено Excel-файлов (*.xlsx / *.xls).")

    for path in files:
        logger.info("Чтение файла %s", path.name)
        parsed = _parse_excel_table(path)
        observations.extend(parsed.observations)
        series_list.append(parsed)

    if not observations:
        logger.error("Не удалось собрать ни одной экспериментальной точки.")
        raise RuntimeError("Не удалось собрать ни одной экспериментальной точки.")
    logger.info("Собрано %d экспериментальных точек из %d файлов", len(observations), len(files))
    return observations, series_list


def _safe_value(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return float(value)


def _residual_vector(param_array: np.ndarray, observations: Sequence[Observation]) -> np.ndarray:
    params = ModelParameters.from_array(param_array)
    residuals: list[float] = []

    for obs in observations:
        try:
            (f_lf, tau_lf), (f_hf, tau_hf) = _predict_single_point(obs.H, obs.T, params)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Пропуск точки (H=%.3f, T=%.3f): ошибка модели: %s", obs.H, obs.T, exc)
            continue

        if not _all_finite(f_lf, tau_lf, f_hf, tau_hf):
            logger.warning("Пропуск точки (H=%.3f, T=%.3f): нечисловой результат модели", obs.H, obs.T)
            continue

        _append_residual(residuals, f_lf, obs.f_lf, obs.err_f_lf, weight=1.0)
        _append_residual(residuals, f_hf, obs.f_hf, obs.err_f_hf, weight=1.0)
        _append_residual(residuals, tau_lf, obs.tau_lf, obs.err_tau_lf, weight=TAU_WEIGHT_FALLBACK)
        _append_residual(residuals, tau_hf, obs.tau_hf, obs.err_tau_hf, weight=TAU_WEIGHT_FALLBACK)

    return np.asarray(residuals, dtype=float)


def _append_residual(
    residuals: list[float],
    model_value: float,
    exp_value: float | None,
    sigma: float | None,
    *,
    weight: float,
) -> None:
    if exp_value is None:
        return
    if not np.isfinite(model_value):
        return
    if sigma is not None and sigma > 0:
        residuals.append((model_value - exp_value) / sigma)
    else:
        residuals.append(weight * (model_value - exp_value))


def fit_parameters(initial: ModelParameters | None = None, data_dir: Path | None = None) -> tuple[ModelParameters, List[ParsedSeries]]:
    data_root = data_dir or DATA_DIR
    observations, parsed_series = _build_observations_and_series(data_root)
    p0 = initial.as_array() if initial else np.array([1.0, 1.0, 1.0, ALPHA_DEFAULT], dtype=float)
    bounds = ([0.1, 0.1, 0.1, 1e-5], [10.0, 10.0, 10.0, 0.05])

    logger.info("Запуск подбора параметров: p0=%s", p0)
    solution = least_squares(_residual_vector, p0, bounds=bounds, args=(observations,), method="trf")
    best = ModelParameters.from_array(solution.x)
    logger.info(
        "Подбор завершён: k_M=%.5f, k_m=%.5f, k_K=%.5f, alpha=%.6f, cost=%.4e, итераций=%d",
        best.k_M,
        best.k_m,
        best.k_K,
        best.alpha,
        solution.cost,
        solution.nfev,
    )
    return best, parsed_series


def _prepare_series(params: ModelParameters, parsed_series: Sequence[ParsedSeries]) -> List[SeriesData]:
    series: List[SeriesData] = []

    for dataset in parsed_series:
        H_values = np.array([obs.H for obs in dataset.observations], dtype=float)
        T_values = np.array([obs.T for obs in dataset.observations], dtype=float)

        model_lf, model_hf, model_lf_tau, model_hf_tau = _evaluate_axis(
            H_values=H_values, T_values=T_values, params=params
        )

        if not _all_finite(model_lf, model_hf, model_lf_tau, model_hf_tau):
            logger.warning("Серия %s: обнаружены нечисловые значения модели, они будут показаны как NaN", dataset.name)

        series.append(
            SeriesData(
                name=dataset.name,
                axis_label=dataset.axis_label,
                axis_values=dataset.axis_values,
                experimental_lf=dataset.exp_f_lf,
                experimental_hf=dataset.exp_f_hf,
                experimental_lf_tau=dataset.exp_tau_lf,
                experimental_hf_tau=dataset.exp_tau_hf,
                model_lf=model_lf,
                model_hf=model_hf,
                model_lf_tau=model_lf_tau,
                model_hf_tau=model_hf_tau,
            )
        )

    return series


def _parse_excel_table(path: Path) -> ParsedSeries:
    """
    Читает Excel-файл с данными по одному срезу (H = const или T = const).

    Ожидаемый формат (см. пример на скриншоте):
        - первая строка: значения оси (H или T), начиная со 2-го столбца;
        - первый столбец: русские названия величин и их погрешностей.
    Имя файла определяет фиксированную величину:  T_<value>.xlsx  или  H_<value>.xlsx.
    """
    df = pd.read_excel(path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"{path.name}: слишком мало столбцов для данных.")

    axis_values = _to_float_array(df.iloc[0, 1:])
    if np.any(~np.isfinite(axis_values)):
        raise ValueError(f"{path.name}: не удалось прочитать значения оси (первая строка).")

    fixed_type, fixed_value = _parse_filename(path.name)
    axis_label = "H (Oe)" if fixed_type == "T" else "T (K)"
    logger.info("Файл %s: фиксировано %s = %s, точек на оси: %d", path.name, fixed_type, fixed_value, len(axis_values))

    rows_map = _extract_rows(df.iloc[1:, :])
    if rows_map.get("f_lf") is None and rows_map.get("f_hf") is None:
        raise ValueError(f"{path.name}: не найдено строк с частотами.")

    exp_f_lf = rows_map.get("f_lf", np.full_like(axis_values, np.nan))
    exp_f_hf = rows_map.get("f_hf", np.full_like(axis_values, np.nan))
    exp_tau_lf = rows_map.get("tau_lf", np.full_like(axis_values, np.nan))
    exp_tau_hf = rows_map.get("tau_hf", np.full_like(axis_values, np.nan))

    err_f_lf = rows_map.get("err_f_lf")
    err_f_hf = rows_map.get("err_f_hf")
    err_tau_lf = rows_map.get("err_tau_lf")
    err_tau_hf = rows_map.get("err_tau_hf")

    observations: list[Observation] = []
    for idx, axis_val in enumerate(axis_values):
        H, T = _axis_to_HT(fixed_type, fixed_value, axis_val)
        observations.append(
            Observation(
                H=H,
                T=T,
                f_lf=_safe_value(exp_f_lf[idx]),
                f_hf=_safe_value(exp_f_hf[idx]),
                tau_lf=_safe_value(exp_tau_lf[idx]),
                tau_hf=_safe_value(exp_tau_hf[idx]),
                err_f_lf=_safe_value(err_f_lf[idx]) if err_f_lf is not None else None,
                err_f_hf=_safe_value(err_f_hf[idx]) if err_f_hf is not None else None,
                err_tau_lf=_safe_value(err_tau_lf[idx]) if err_tau_lf is not None else None,
                err_tau_hf=_safe_value(err_tau_hf[idx]) if err_tau_hf is not None else None,
            )
        )

    logger.info(
        "Файл %s: считано точек %d (f_lf: %s, f_hf: %s)",
        path.name,
        len(observations),
        "да" if rows_map.get("f_lf") is not None else "нет",
        "да" if rows_map.get("f_hf") is not None else "нет",
    )
    return ParsedSeries(
        name=path.stem,
        axis_label=axis_label,
        axis_values=axis_values,
        observations=observations,
        exp_f_lf=exp_f_lf,
        exp_f_hf=exp_f_hf,
        exp_tau_lf=exp_tau_lf,
        exp_tau_hf=exp_tau_hf,
    )


def _axis_to_HT(fixed_type: str, fixed_value: float, axis_value: float) -> tuple[float, float]:
    if fixed_type == "T":  # файл T_<value>: температура фиксирована, ось = H
        return float(axis_value), float(fixed_value)
    else:  # fixed_type == "H": поле фиксировано, ось = T
        return float(fixed_value), float(axis_value)


def _parse_filename(name: str) -> tuple[str, float]:
    stem = Path(name).stem
    if stem.upper().startswith("T_"):
        return "T", float(stem.split("_", 1)[1])
    if stem.upper().startswith("H_"):
        return "H", float(stem.split("_", 1)[1])
    raise ValueError(f"{name}: имя файла должно начинаться с T_<val> или H_<val>.")


def _extract_rows(df_body: pd.DataFrame) -> dict[str, np.ndarray]:
    """Извлекает числовые ряды из таблицы по текстовым меткам."""
    label_map = {
        "частотанчггц": "f_lf",
        "частотавчггц": "f_hf",
        "времязатуханиянчнс": "tau_lf",
        "времязатуханиявчнс": "tau_hf",
        "погрчастотанчггц": "err_f_lf",
        "погрчастотавчггц": "err_f_hf",
        "погрвремязатуханиянчнс": "err_tau_lf",
        "погрвремязатуханиявчнс": "err_tau_hf",
        # варианты с точкой после "погр" (из точных названий строк)
        "погр.частотанчггц": "err_f_lf",
        "погр.частотавчггц": "err_f_hf",
        "погр.времязатуханиянчнс": "err_tau_lf",
        "погр.времязатуханиявчнс": "err_tau_hf",
    }
    result: dict[str, np.ndarray] = {}
    for _, row in df_body.iterrows():
        label_raw = str(row.iloc[0])
        key = _normalize_label(label_raw)
        mapped = label_map.get(key)
        if mapped is None:
            continue
        result[mapped] = _to_float_array(row.iloc[1:])
    return result


def _normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    normalized = normalized.replace("ё", "е")
    for ch in [" ", ".", ",", "\t"]:
        normalized = normalized.replace(ch, "")
    return normalized


def _to_float_array(series: pd.Series | Sequence) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return arr


def _all_finite(*values: np.ndarray | float) -> bool:
    return all(np.all(np.isfinite(v)) for v in values)


def _evaluate_axis(H_values: Iterable[float], T_values: Iterable[float], params: ModelParameters):
    H_values = np.asarray(H_values, dtype=float)
    T_values = np.asarray(T_values, dtype=float)
    lf_freq, hf_freq = [], []
    lf_tau, hf_tau = [], []
    for H, T in zip(H_values, T_values):
        try:
            (f_lf, tau_lf), (f_hf, tau_hf) = _predict_single_point(float(H), float(T), params)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Серия: пропуск точки (H=%.3f, T=%.3f) из-за ошибки модели: %s", H, T, exc)
            f_lf = f_hf = tau_lf = tau_hf = np.nan
        lf_freq.append(f_lf)
        hf_freq.append(f_hf)
        lf_tau.append(tau_lf)
        hf_tau.append(tau_hf)
    return (np.asarray(lf_freq), np.asarray(hf_freq), np.asarray(lf_tau), np.asarray(hf_tau))


def _materials_at_temperature(temperature: float) -> Tuple[float, float, float]:
    """
    Возвращает (m, M, K) для заданной температуры.

    Интерполяция не используется: берётся ближайший узел из T_VALS.
    """
    idx = np.argmin(np.abs(T_VALS - temperature))
    if not np.isclose(T_VALS[idx], temperature, atol=1e-6):
        logger.warning("Температура %.3f K отсутствует в сетке T_VALS, взят ближайший узел %.3f K", temperature, T_VALS[idx])
    return float(m_ARRAY[idx]), float(M_ARRAY[idx]), float(K_ARRAY[idx])


def main(data_dir: str | None = None) -> None:
    configure_logging()
    best, parsed_series = fit_parameters(data_dir=Path(data_dir) if data_dir else None)
    logger.info(
        "Оптимальные параметры: k_M=%.4f, k_m=%.4f, k_K=%.4f, alpha=%.6f",
        best.k_M,
        best.k_m,
        best.k_K,
        best.alpha,
    )

    series = _prepare_series(best, parsed_series)
    fig = build_summary_figure(series)
    fig.show()
    logger.info("Готово. График открыт в браузере.")


if __name__ == "__main__":
    main()
