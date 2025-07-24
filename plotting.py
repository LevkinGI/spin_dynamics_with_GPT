# plotting.py
import plotly.graph_objs as go
import numpy as np

def create_phi_fig(time, phi, phi_fit, H, T, approx_freqs_GHz, theor_freqs_GHz, material):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=phi, mode='lines', name='Численное решение', line=dict(color='red')))
    if phi_fit is not None:
        fig.add_trace(go.Scatter(x=time, y=phi_fit, mode='lines', name='Аппроксимация', line=dict(color='blue', dash='dash')))
    fig.update_layout(
        title="Динамика угла φ",
        xaxis_title="Время (нс)",
        yaxis_title="Отклонение угла φ (градусы)",
        template="plotly_white",
        annotations=[{
            "x": 0.95,
            "y": 1.05,
            "text": (
                f"<b>Материал</b> {'FeFe' if material == '1' else 'GdFe'}<br>"
                f"<b>Магнитное поле</b> H = {H} Э<br>"
                f"<b>Температура</b> T = {T} K<br>"
                f"<b>HF</b> Аппроксимация: {approx_freqs_GHz[0]} ГГц; Аналитика: {theor_freqs_GHz[0]} ГГц<br>"
                f"<b>LF</b> Аппроксимация: {approx_freqs_GHz[1]} ГГц; Аналитика: {theor_freqs_GHz[1]} ГГц"
            ),
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
            "bordercolor": "black",
            "borderwidth": 1,
            "borderpad": 10,
            "bgcolor": "white"
        }],
        font=dict(size=18)
    )
    return fig

def create_theta_fig(time, theta, theta_fit):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=theta, mode='lines', name='Численное решение', line=dict(color='red')))
    if theta_fit is not None:
        fig.add_trace(go.Scatter(x=time, y=theta_fit, mode='lines', name='Аппроксимация', line=dict(color='blue', dash='dash')))
    fig.update_layout(
        title="Динамика угла θ",
        xaxis_title="Время (нс)",
        yaxis_title="Отклонение угла θ (градусы)",
        template="plotly_white",
        font=dict(size=18)
    )
    return fig

def create_yz_fig(y, z, time, anim_speed=5):
    frames = []
    num_frames = len(time[::anim_speed])
    for i in range(num_frames):
        frame = go.Frame(
            data=[go.Scatter(
                x=y[: (i+1) * anim_speed],
                y=z[: (i+1) * anim_speed],
                mode='lines',
                line=dict(color='rgb(255, 0, 0)', width=2)
            )]
        )
        frames.append(frame)
    fig = go.Figure(
        data=[go.Scatter(x=y, y=z, mode='lines', line=dict(color='rgb(255, 0, 0)', width=2))],
        frames=frames
    )
    lim = np.max([np.abs(y), np.abs(z)])
    limits = (-1.1 * lim, 1.1 * lim)
    fig.update_layout(
        title="Проекция траектории на плоскость yz",
        xaxis_title="Координата y (ед. L)",
        yaxis_title="Координата z (ед. L)",
        xaxis=dict(range=limits),
        yaxis=dict(range=limits),
        template="plotly_white",
        updatemenus=[{
            "type": "buttons",
            "showactive": True,
            "x": 1,
            "y": 1,
            "xanchor": 'right',
            "yanchor": 'top',
            "direction": 'left',
            "buttons": [{
                "label": "Запуск",
                "method": "animate",
                "args": [None, {"frame": {"duration": 50, "redraw": True},
                                "fromcurrent": False, "mode": "immediate"}]
            }]
        }],
        font=dict(size=18)
    )
    return fig

def create_H_fix_fig(T_vals, H_fix_freqs, H, data=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T_vals, y=H_fix_freqs[0], mode='lines', name='HF', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=T_vals, y=H_fix_freqs[1], mode='lines', name='LF', line=dict(color='red')))
    if data is not None:
        fig.add_trace(go.Scatter(x=data[0], y=data[1], mode='markers', name='LF', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=data[0], y=data[2], mode='markers', name='HF', line=dict(color='blue')))
    fig.update_layout(
        title=f"Зависимость частот от температуры при H = {H} Э",
        xaxis_title="Температура (K)",
        yaxis_title="Частота (ГГц)",
        font=dict(size=18),
        template="plotly_white"
    )
    return fig

def create_T_fix_fig(H_vals, T_fix_freqs, T, data=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=H_vals, y=T_fix_freqs[0], mode='lines', name='HF', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=H_vals, y=T_fix_freqs[1], mode='lines', name='LF', line=dict(color='red')))
    if data is not None:
        fig.add_trace(go.Scatter(x=data[0], y=data[1], mode='markers', name='LF', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=data[0], y=data[2], mode='markers', name='HF', line=dict(color='blue')))
    fig.update_layout(
        title=f"Зависимость частот от магнитного поля при T = {T} K",
        xaxis_title="Магнитное поле (Э)",
        yaxis_title="Частота (ГГц)",
        font=dict(size=18),
        template="plotly_white"
    )
    return fig

def create_phi_amp_fig(T_vals, H_vals, amplitude_phi_static):
    fig = go.Figure(
        data=[go.Surface(
            z=amplitude_phi_static,
            x=H_vals,
            y=T_vals,
            colorscale=[[0, 'rgb(173, 216, 230)'], [1, 'rgb(0, 0, 255)']],
            showscale=False,
            name='LF'
        )],
        layout=go.Layout(
            scene=dict(
                xaxis_title='Магнитное поле (Э)',
                yaxis_title='Температура (K)',
                zaxis_title='Амплитуда φ (°)'
            ),
            font=dict(size=14),
            template="plotly_white"
        )
    )
    return fig

def create_theta_amp_fig(T_vals, H_vals, amplitude_theta_static):
    fig = go.Figure(
        data=[go.Surface(
            z=amplitude_theta_static,
            x=H_vals,
            y=T_vals,
            colorscale=[[0, 'rgb(255, 182, 193)'], [1, 'rgb(255, 0, 0)']],
            showscale=False,
            name='HF'
        )],
        layout=go.Layout(
            scene=dict(
                xaxis_title='Магнитное поле (Э)',
                yaxis_title='Температура (K)',
                zaxis_title='Амплитуда θ (°)'
            ),
            font=dict(size=14),
            template="plotly_white"
        )
    )
    return fig

def create_freq_fig(T_vals, H_vals, freq_array1, freq_array2):
    fig = go.Figure(
        data=[
            go.Surface(z=freq_array1, x=H_vals, y=T_vals,
                       colorscale=[[0, 'rgb(255, 182, 193)'], [1, 'rgb(255, 0, 0)']],
                       showscale=False, name='HF'),
            go.Surface(z=freq_array2, x=H_vals, y=T_vals,
                       colorscale=[[0, 'rgb(173, 216, 230)'], [1, 'rgb(0, 0, 255)']],
                       showscale=False, name='LF')
        ],
        layout=go.Layout(
            title="Частоты LF и HF в зависимости от H и T",
            scene=dict(
                xaxis_title='Магнитное поле (Э)',
                yaxis_title='Температура (K)',
                zaxis_title='Частота (ГГц)'
            ),
            font=dict(size=14),
            template="plotly_white"
        )
    )
    return fig

def create_phase_fig(T_vals, H_vals, theta_0):
    theta_0 = theta_0.T
    custom_colorscale = [
        [0.00, 'rgb(0, 0, 0)'],        # black
        [0.31, 'rgb(0, 0, 255)'],      # blue
        [0.62, 'rgb(0, 128, 0)'],      # green
        [0.93, 'rgb(255, 255, 0)'],    # yellow
        [1.00, 'rgb(255, 255, 255)']   # white
    ]
    heat = go.Heatmap(
        x=T_vals,
        y=H_vals,
        z=theta_0,
        colorscale=custom_colorscale,
        colorbar=dict(
            title=dict(
                text=r'θ₀ (rad)',
                side='right'
            ),
            tickmode='array',
            tickvals=[0, np.pi / 4, np.pi / 2],
            ticktext=['0', 'π/4', 'π/2'],
            outlinewidth=1
        ),
        zmin=0, zmax=np.pi / 2,
        showscale=True
    )

    fig = go.Figure(data=[heat])

    contour = go.Contour(
        x=T_vals,
        y=H_vals,
        z=theta_0,
        showscale=False,
        contours=dict(
            start=0.01, end=0.01, size=0.01,
            coloring='none'
        ),
        line=dict(width=1.5, color='white')
    )
    fig.add_trace(contour)

    mask = (theta_0[:, 0] > 0) & (theta_0[:, 0] < 0.1)
    if np.any(mask):
        idx_min, idx_max = np.where(mask)[0][[0, -1]]
        y_noncol = H_vals[idx_max] + 300
        y_col    = H_vals[idx_min] - 300
        fig.add_annotation(x=T_vals[0], y=y_noncol,
                           text='non‑collinear',
                           showarrow=False, font=dict(color='white', size=14),
                           xanchor='left', yanchor='bottom')
        fig.add_annotation(x=T_vals[0], y=y_col,
                           text='collinear',
                           showarrow=False, font=dict(color='white', size=14),
                           xanchor='left', yanchor='top')

    fig.update_layout(
        xaxis=dict(title='T (K)', range=[T_vals.min(), T_vals.max()]),
        yaxis=dict(title='H (Oe)', range=[H_vals.min(), H_vals.max()]),
        template='plotly_white',
        margin=dict(l=60, r=40, t=40, b=60)
    )

    return fig

    
__all__ = [
    'create_phi_fig', 'create_theta_fig', 'create_yz_fig', 'create_H_fix_fig', 'create_phase_fig',
    'create_T_fix_fig', 'create_phi_amp_fig', 'create_theta_amp_fig', 'create_freq_fig',
]
