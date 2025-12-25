"""Визуализация результатов аппроксимации."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FIT_LF = "#e74c3c"
FIT_HF = "#1f77b4"
MARKER_SIZE = 9


@dataclass
class SeriesData:
    name: str
    axis_label: str
    exp_axis: np.ndarray
    experimental_lf: np.ndarray
    experimental_hf: np.ndarray
    experimental_lf_tau: np.ndarray
    experimental_hf_tau: np.ndarray
    model_axis: np.ndarray
    model_lf: np.ndarray
    model_hf: np.ndarray
    model_lf_tau: np.ndarray
    model_hf_tau: np.ndarray


def _scatter_pair(fig, row: int, col: int, model_axis, model, exp_axis, exp, label_base: str, color: str):
    fig.add_trace(
        go.Scatter(x=model_axis, y=model, mode="lines", line=dict(width=2, color=color), name=f"{label_base} model"),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=exp_axis,
            y=exp,
            mode="markers",
            marker=dict(size=MARKER_SIZE, color=color, line=dict(width=1, color="#000000")),
            name=f"{label_base} exp",
        ),
        row=row,
        col=col,
    )


def build_summary_figure(series: Sequence[SeriesData]):
    """
    Строит сводный график (частоты и времена затухания) для всех серий.

    Первая строка — частоты LF/HF, вторая — времена затухания.
    Колонки соответствуют элементам списка *series*.
    """
    if not series:
        raise ValueError("Нет данных для построения графика.")

    cols = len(series)
    titles = [s.name for s in series]
    fig = make_subplots(
        rows=2,
        cols=cols,
        vertical_spacing=0.08,
        horizontal_spacing=0.04,
        shared_xaxes=False,
        subplot_titles=tuple(titles),
    )

    for idx, s in enumerate(series, start=1):
        exp_axis = s.exp_axis
        model_axis = s.model_axis
        _scatter_pair(fig, 1, idx, model_axis, s.model_lf, exp_axis, s.experimental_lf, "f_LF", FIT_LF)
        _scatter_pair(fig, 1, idx, model_axis, s.model_hf, exp_axis, s.experimental_hf, "f_HF", FIT_HF)
        _scatter_pair(fig, 2, idx, model_axis, s.model_lf_tau, exp_axis, s.experimental_lf_tau, "tau_LF", FIT_LF)
        _scatter_pair(fig, 2, idx, model_axis, s.model_hf_tau, exp_axis, s.experimental_hf_tau, "tau_HF", FIT_HF)

        fig.update_xaxes(title_text=s.axis_label, row=2, col=idx)
        for row in (1, 2):
            fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True, showgrid=True, gridcolor="#cccccc", gridwidth=1, row=row, col=idx)
            fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True, showgrid=True, gridcolor="#cccccc", gridwidth=1, row=row, col=idx)

    fig.update_yaxes(title_text="Частота (ГГц)", row=1, col=1)
    fig.update_yaxes(title_text="Время затухания (нс)", row=2, col=1)
    fig.update_layout(
        showlegend=False,
        hovermode="x unified",
        font=dict(size=16),
        width=max(400 * cols, 1200),
        height=800,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=20)
    fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=16))
    fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=16))
    return fig


__all__ = ["SeriesData", "build_summary_figure"]
