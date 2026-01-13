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
    experimental_lf_err: np.ndarray | None = None
    experimental_hf_err: np.ndarray | None = None
    experimental_lf_tau_err: np.ndarray | None = None
    experimental_hf_tau_err: np.ndarray | None = None


@dataclass
class PhaseDiagramData:
    temp_mesh: np.ndarray
    field_mesh: np.ndarray
    theta_exp: np.ndarray
    theta_model: np.ndarray
    temp_label: str
    field_label: str


def _axis_from_mesh(mesh: np.ndarray, axis: int) -> np.ndarray:
    if mesh.ndim != 2:
        return np.unique(mesh)
    if axis == 0:
        values = mesh[:, 0]
    else:
        values = mesh[0, :]
    return np.asarray(values, dtype=float)


def _scatter_pair(
    fig,
    row: int,
    col: int,
    model_axis,
    model,
    exp_axis,
    exp,
    label_base: str,
    color: str,
    error: np.ndarray | None = None,
):
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
            error_y=dict(type="data", array=error, visible=error is not None),
            marker=dict(size=MARKER_SIZE, color=color, line=dict(width=1, color="#000000")),
            name=f"{label_base} exp",
        ),
        row=row,
        col=col,
    )


def build_summary_figure(series: Sequence[SeriesData], phase_diagram: PhaseDiagramData | None = None):
    """
    Строит сводный график (частоты и времена затухания) для всех серий.

    Первая строка — частоты LF/HF, вторая — времена затухания.
    Колонки соответствуют элементам списка *series*.
    """
    if not series:
        raise ValueError("Нет данных для построения графика.")

    cols = len(series) + (1 if phase_diagram is not None else 0)
    titles: list[str] = []
    for row in range(2):
        for col in range(cols):
            if phase_diagram is not None and col == 0:
                titles.append("эксперимент" if row == 0 else "аппроксимация")
            elif row == 0:
                series_idx = col - (1 if phase_diagram is not None else 0)
                titles.append(series[series_idx].name)
            else:
                titles.append("")
    fig = make_subplots(
        rows=2,
        cols=cols,
        vertical_spacing=0.08,
        horizontal_spacing=0.04,
        shared_xaxes=False,
        subplot_titles=tuple(titles),
    )

    col_offset = 0
    if phase_diagram is not None:
        temp_axis = _axis_from_mesh(phase_diagram.temp_mesh, axis=1)
        field_axis = _axis_from_mesh(phase_diagram.field_mesh, axis=0)
        theta_min = np.nanmin([np.nanmin(phase_diagram.theta_exp), np.nanmin(phase_diagram.theta_model)])
        theta_max = np.nanmax([np.nanmax(phase_diagram.theta_exp), np.nanmax(phase_diagram.theta_model)])
        fig.add_trace(
            go.Heatmap(
                x=temp_axis,
                y=field_axis,
                z=phase_diagram.theta_exp,
                colorscale="Viridis",
                zmin=theta_min,
                zmax=theta_max,
                showscale=False,
                name="theta_exp",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                x=temp_axis,
                y=field_axis,
                z=phase_diagram.theta_model,
                colorscale="Viridis",
                zmin=theta_min,
                zmax=theta_max,
                showscale=False,
                name="theta_model",
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text=phase_diagram.temp_label, row=2, col=1)
        fig.update_yaxes(title_text=phase_diagram.field_label, row=1, col=1)
        fig.update_yaxes(title_text=phase_diagram.field_label, row=2, col=1)
        for row in (1, 2):
            fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True, showgrid=True, gridcolor="#cccccc", gridwidth=1, row=row, col=1)
            fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True, showgrid=True, gridcolor="#cccccc", gridwidth=1, row=row, col=1)
        col_offset = 1

    for idx, s in enumerate(series, start=1 + col_offset):
        exp_axis = s.exp_axis
        model_axis = s.model_axis
        _scatter_pair(
            fig,
            1,
            idx,
            model_axis,
            s.model_lf,
            exp_axis,
            s.experimental_lf,
            "f_LF",
            FIT_LF,
            error=s.experimental_lf_err,
        )
        _scatter_pair(
            fig,
            1,
            idx,
            model_axis,
            s.model_hf,
            exp_axis,
            s.experimental_hf,
            "f_HF",
            FIT_HF,
            error=s.experimental_hf_err,
        )
        _scatter_pair(
            fig,
            2,
            idx,
            model_axis,
            s.model_lf_tau,
            exp_axis,
            s.experimental_lf_tau,
            "tau_LF",
            FIT_LF,
            error=s.experimental_lf_tau_err,
        )
        _scatter_pair(
            fig,
            2,
            idx,
            model_axis,
            s.model_hf_tau,
            exp_axis,
            s.experimental_hf_tau,
            "tau_HF",
            FIT_HF,
            error=s.experimental_hf_tau_err,
        )

        fig.update_xaxes(title_text=s.axis_label, row=2, col=idx)
        for row in (1, 2):
            fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True, showgrid=True, gridcolor="#cccccc", gridwidth=1, row=row, col=idx)
            fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True, showgrid=True, gridcolor="#cccccc", gridwidth=1, row=row, col=idx)

    freq_col = 1 + col_offset
    fig.update_yaxes(title_text="Частота (ГГц)", row=1, col=freq_col)
    fig.update_yaxes(title_text="Время затухания (нс)", row=2, col=freq_col)
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


__all__ = ["PhaseDiagramData", "SeriesData", "build_summary_figure"]
