# app.py

# Автоматическая сборка Cython модулей при импорте
import pyximport
import numpy as np
import time
import io
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)

import dash
from dash import dcc, html, no_update, callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_daq as daq
import plotly.graph_objs as go
from scipy.optimize import least_squares

from dataclasses import asdict
from config import SimParams
from constants import *
from simulator import run_simulation
# Функции аппроксимации из Cython-модуля:
from fitting_cy import fit_function_theta, fit_function_phi
from fitting import residuals_stage1, residuals_stage2, combined_residuals
from plotting import *

sliders_range = 5
log_marks = {}
for i in  range(1, sliders_range+1):
    if i > 10 and i % 10 != 0: continue
    log_marks[np.log10(i)]  = str(i)
    log_marks[-np.log10(i)] = '1/'+str(i)

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    dcc.Store(
        id='param-store',
        data={
            "1": asdict(SimParams(1.0, 1.0, 1.0, 1.0)),
            "2": asdict(SimParams(1.0, 1.0, 1.0, 1.0)),
        }
    ),

    html.H1("Динамика углов θ и φ при различных значениях магнитного поля и температуры"),
    html.Label(id='H-label'),
    dcc.Slider(
        id='H-slider',
        min=0,
        max=H_vals[-1],
        step=10,
        value=1000,
        marks={i: str(i) for i in range(0, H_vals[-1] + 1, 500)},
        tooltip={"placement": "bottom", "always_visible": False}, updatemode="mouseup",
    ),
    html.Div(id='selected-H-value', style={'margin-bottom': '20px'}),
    html.Label(id='T-label'),
    dcc.Slider(
        id='T-slider',
        min=290,
        max=350,
        step=0.1,
        value=T_init,
        marks={i: str(i) for i in range(290, 351, 10)},
        tooltip={"placement": "bottom", "always_visible": False}, updatemode="mouseup",
    ),
    html.Div(id='selected-T-value', style={'margin-bottom': '20px'}),




    
    html.Div([
        html.Div([
            html.Label(id='alpha-scale-label'),
            dcc.Slider(id='alpha-scale-slider',
                       min=-np.log10(sliders_range), max=np.log10(sliders_range), step=0.001, value=0.0,
                       marks=log_marks,
                       updatemode="mouseup",
                       vertical=True, verticalHeight=180,
                      ),
            ],
            style={"marginLeft": "30px","position": "relative"}
            ),
        
        html.Div([
            html.Label(id='chi-scale-label'),
            dcc.Slider(id='chi-scale-slider',
                       min=-np.log10(sliders_range), max=np.log10(sliders_range), step=0.001, value=0.0,
                       marks=log_marks,
                       updatemode="mouseup",
                       vertical=True, verticalHeight=180,
                      ),
            ],
            style={"marginLeft": "30px", "position": "relative"}
        ),
        
        html.Div([
            html.Label(id='k-scale-label'),
            dcc.Slider(id='k-scale-slider',
                       min=-np.log10(sliders_range), max=np.log10(sliders_range), step=0.001, value=0.0,
                       marks=log_marks,
                       updatemode="mouseup",
                       vertical=True, verticalHeight=180,
                      ),
            ],
            style={"marginLeft": "30px", "position": "relative"}
        ),
        
        html.Div([
            html.Label(id='m-scale-label'),
            dcc.Slider(id='m-scale-slider',
                       min=-np.log10(sliders_range), max=np.log10(sliders_range), step=0.001, value=0.0,
                       marks=log_marks,
                       updatemode="mouseup",
                       vertical=True, verticalHeight=180,
                      ),
            ],
            style={"marginLeft": "30px", "position": "relative"}
        ),
        daq.BooleanSwitch(
            id='auto-calc-switch',
            on=False,
            label='Моделирование при изменении параметров',
            labelPosition='top',
            color='#119DFF',
            style={"marginLeft": "60px"}
        ),
        ],
        style={
            "display":   "flex",
            "alignItems": "flex-start",   # вершины всех ползунков выровнены
            "flexWrap":  "nowrap",         # гарантирует одну строку
        },
    ),




    
    dcc.Dropdown(
        id='material-dropdown',
        options=[
            {'label': 'FeFe', 'value': '1'},
            {'label': 'GdFe', 'value': '2'}
        ],
        value='1',
        style={'width': '300px'}
    ),


    

    
    html.Div([
        dcc.Graph(id='phase-graph', style={'display': 'inline-block', 'verticalAlign': 'top',
                                           'width': '25%', 'height': 'calc(19vw)'}),
        dcc.Graph(
            id='frequency-surface-graph',
            style={'display': 'inline-block', 'verticalAlign': 'top',
                   'width': '25%', 'height': 'calc(25vw)'},
            figure=go.Figure(
                data=[
                    go.Surface(z=f1_GHz, x=H_vals, y=T_vals_1,
                               colorscale=[[0, 'rgb(255, 182, 193)'], [1, 'rgb(255, 0, 0)']],
                               showscale=False, name='HF'),
                    go.Surface(z=f2_GHz, x=H_vals, y=T_vals_1,
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
        ),
        html.Div([
            html.Button('Скачать данные', id='download-H-btn',
                        style={'margin-bottom': '5px'}),
            dcc.Download(id='download-H-file'),
            dcc.Graph(id='H_fix-graph'),
        ], style={'display': 'inline-block', 'verticalAlign': 'top',
                  'width': '25%', 'height': 'calc(25vw)'}),
        html.Div([
            html.Button('Скачать данные', id='download-T-btn',
                        style={'margin-bottom': '5px'}),
            dcc.Download(id='download-T-file'),
            dcc.Graph(id='T_fix-graph'),
        ], style={'display': 'inline-block', 'verticalAlign': 'top',
                  'width': '25%', 'height': 'calc(25vw)'}),
    ]),
                      



    
    html.Div([
        dcc.Graph(id='phi-graph', style={'display': 'inline-block', 'width': '50%'}),
        dcc.Graph(id='theta-graph', style={'display': 'inline-block', 'width': '50%'})
    ]),
    html.Div([
        dcc.Graph(
            id='phi-amplitude-graph',
            style={'display': 'inline-block', 'width': '33%', 'height': 'calc(33vw)'},
            figure=go.Figure(
                data=[go.Surface(z=phi_amplitude, x=H_vals, y=T_vals_1,
                                 colorscale=[[0, 'rgb(173, 216, 230)'], [1, 'rgb(0, 0, 255)']],
                                 showscale=False, name='LF')],
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
        ),
        dcc.Graph(
            id='theta-amplitude-graph',
            style={'display': 'inline-block', 'width': '33%', 'height': 'calc(33vw)'},
            figure=go.Figure(
                data=[go.Surface(z=theta_amplitude, x=H_vals, y=T_vals_1,
                                 colorscale=[[0, 'rgb(255, 182, 193)'], [1, 'rgb(255, 0, 0)']],
                                 showscale=False, name='HF')],
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
        ),
        dcc.Graph(id='yz-graph', style={'display': 'inline-block', 'width': '33%', 'height': 'calc(33vw)'})
    ]),
])

@app.callback(
    [Output('H-label', 'children'),
     Output('T-label', 'children')],
    [Input('H-slider', 'value'),
     Input('T-slider', 'value')],
)
def update_slider_values(H, T):
    return f'Магнитное поле H = {H} Э:', f'Температура T = {T} K:'

@app.callback(
    [Output('T-slider', 'min'),
     Output('T-slider', 'max'),
     Output('T-slider', 'step'),
     Output('T-slider', 'value'),
     Output('T-slider', 'marks')],
    [Input('material-dropdown', 'value')],
    [State('T-slider', 'value')],
    prevent_initial_call=True,
)
def update_T_slider(material, T):
    if material == '1':
        t_vals = T_vals_1
    else:
        t_vals = T_vals_2
    if T is None:
        T = T_init
    t_index = np.abs(t_vals - T).argmin()
    
    min_val = t_vals[0]
    max_val = t_vals[-1]
    step = np.round(t_vals[1] - t_vals[0], decimals=1) 
    value = t_vals[t_index]
    marks = {float(val): str(val) for val in t_vals if val % 10 == 0}
    return min_val, max_val, step, value, marks
    
@app.callback(
    Output("alpha-scale-label", "children"),
    Input("alpha-scale-slider", "drag_value"),
    prevent_initial_call=True,
)
def move_alpha_slider(logk):
    if logk is None:                          # при первом рендере drag_value == None
        raise PreventUpdate
    k = 10**logk
    return f"{k:.2f} × α"

@app.callback(
    Output("chi-scale-label", "children"),
    Input("chi-scale-slider", "drag_value"),
    prevent_initial_call=True,
)
def move_chi_slider(logk):
    if logk is None:
        raise PreventUpdate
    k = 10**logk
    return f"{k:.2f} × χ"

@app.callback(
    Output("k-scale-label", "children"),
    Input("k-scale-slider", "drag_value"),
    prevent_initial_call=True,
)
def move_k_slider(logk):
    if logk is None:
        raise PreventUpdate
    k = 10**logk
    return f"{k:.2f} × K(T)"

@app.callback(
    Output("m-scale-label", "children"),
    Input("m-scale-slider", "drag_value"),
    prevent_initial_call=True,
)
def move_m_slider(logk):
    if logk is None:
        raise PreventUpdate
    k = 10**logk
    return f"{k:.2f} × m"

@app.callback(
    [Output('H_fix-graph', 'figure'),
     Output('T_fix-graph', 'figure'),
     Output('phase-graph', 'figure')],
    [Input('H-slider', 'value'),
    Input('T-slider', 'value'),
    Input("chi-scale-slider", "value"),
    Input("k-scale-slider", "value"),
    Input("m-scale-slider", "value"),
    Input('material-dropdown', 'value')],
)
def live_fix_graphs(H, T, chi_val, k_val, m_val, material):
    chi_scale = 10**chi_val
    k_scale   = 10**k_val
    m_scale   = 10**m_val
    
    T_vals    = T_vals_1 if material == '1' else T_vals_2
    t_index   = np.abs(T_vals - T).argmin()
    m_vec_T   = m_scale * (m_array_1 if material == '1' else m_array_2)
    K_vec_T   = k_scale * (K_array_1 if material == '1' else K_array_2)
    chi_vec_T = chi_scale * (chi_array_1 if material == '1' else chi_array_2)

    m_T   = m_vec_T[t_index]
    K_T   = K_vec_T[t_index]
    chi_T = chi_vec_T[t_index]

    f1_T, f2_T = compute_frequencies_H_fix(H, m_vec_T, chi_vec_T, K_vec_T, gamma)
    f1_H, f2_H = compute_frequencies_T_fix(H_vals, m_T, chi_T, K_T, gamma)
    H_data = H_1000 if H==1000 and material == '1' else None
    T_data = T_293 if T==293 and material == '1' else None

    H_mesh = H_mesh_1 if material == '1' else H_mesh_2
    m_mesh = m_scale * (m_mesh_1 if material == '1' else m_mesh_2)
    K_mesh = k_scale * (K_mesh_1 if material == '1' else K_mesh_2)
    chi_mesh = chi_scale * (chi_mesh_1 if material == '1' else chi_mesh_2)
    theta_0 = compute_phases(H_mesh, m_mesh, K_mesh, chi_mesh)

    H_fix_fig = create_H_fix_fig(T_vals, (f1_T, f2_T), H, H_data)
    T_fix_fig = create_T_fix_fig(H_vals, (f1_H, f2_H), T, T_data)
    phase_fig = create_phase_fig(T_vals, H_vals, theta_0)

    return H_fix_fig, T_fix_fig, phase_fig

@app.callback(
    [Output('alpha-scale-slider',      'value'),
    Output('chi-scale-slider',      'value'),
    Output('k-scale-slider',      'value'),
    Output('m-scale-slider',      'value')],
    Input('material-dropdown', 'value'),
    State('param-store',       'data'),   
    prevent_initial_call=True, 
)
def sync_sliders_with_material(material, store):
    p = SimParams(**store[material])
    return (np.log10(p.alpha_scale), np.log10(p.chi_scale),
            np.log10(p.k_scale), np.log10(p.m_scale))

@app.callback(
    Output('param-store', 'data'),
    [Input('material-dropdown', 'value'),
    Input('alpha-scale-slider',      'value'),
    Input('chi-scale-slider',        'value'),
    Input('k-scale-slider',    'value'),
    Input('m-scale-slider',    'value')],
    State('param-store',       'data'),
    prevent_initial_call=True,
)
def update_params(material, a_k, chi_k, k_k, m_k, store):
    p = SimParams(**store[material])
    p.alpha_scale = 10 ** a_k
    p.chi_scale = 10 ** chi_k
    p.k_scale = 10 ** k_k
    p.m_scale = 10 ** m_k
    store[material] = asdict(p)
    return store

@app.callback(
    [Output('phi-graph', 'figure'),
     Output('theta-graph', 'figure'),
     Output('yz-graph', 'figure'),
     Output('phi-amplitude-graph', 'figure'),
     Output('theta-amplitude-graph', 'figure'),
     Output('frequency-surface-graph', 'figure')],
    [Input('param-store', 'data'),
     Input('H-slider', 'value'),
     Input('T-slider', 'value'),
     Input('material-dropdown', 'value'),
     Input('auto-calc-switch',  'on')],
    prevent_initial_call=True,
)
def update_graphs(store, H, T, material, calc_on):
    if not calc_on: raise PreventUpdate
        
    # Определяем, какой input вызвал callback
    ctx = callback_context
    triggered_inputs = [t['prop_id'] for t in ctx.triggered]
    material_changed = any('material-dropdown' in ti for ti in triggered_inputs)
    params_changed   = any('param-store' in ti for ti in triggered_inputs)
    switch_on = any('auto-calc-switch' in ti for ti in triggered_inputs)
        
    p = SimParams(**store[material])
  
    h_index = np.abs(H_vals - H).argmin()
    
    # Выбор данных в зависимости от материала
    T_vals = T_vals_1 if material=='1' else T_vals_2
    t_index = np.abs(T_vals - T).argmin()
    m_val = p.m_scale * (m_array_1 if material=='1' else m_array_2)[t_index]
    M_val = (M_array_1 if material=='1' else M_array_2)[t_index]
    chi_val = p.chi_scale * (chi_array_1 if material=='1' else chi_array_2)[t_index]
    K_val = p.k_scale * (K_array_1 if material=='1' else K_array_2)[t_index]
    alpha = p.alpha_scale * (alpha_1 if material=='1' else alpha_2)
    amplitude_phi_static = phi_amplitude if material=='1' else phi_amplitude_2
    amplitude_theta_static = theta_amplitude if material=='1' else theta_amplitude_2
    kappa = m_val / gamma

    H_mesh = H_mesh_1 if material == '1' else H_mesh_2
    m_mesh = p.m_scale * (m_mesh_1 if material == '1' else m_mesh_2)
    K_mesh = p.k_scale * (K_mesh_1 if material == '1' else K_mesh_2)
    chi_mesh = p.chi_scale * (chi_mesh_1 if material == '1' else chi_mesh_2)
    freq_array1, freq_array2 = compute_frequencies(H_mesh, m_mesh, chi_mesh, K_mesh, gamma)
    theor_freqs_GHz = sorted(np.round([freq_array1[t_index, h_index], freq_array2[t_index, h_index]], 1), reverse=True)

    sim_time, sol = run_simulation(H, m_val, M_val, K_val, chi_val, alpha, kappa)

    time_ns = sim_time * 1e9
    theta = np.degrees(sol[0])
    phi = np.degrees(sol[1])

    # Выполнение аппроксимации
    if False:
        A1_theta = np.max(theta) / 2
        A2_theta = A1_theta
        A1_phi = np.max(phi) / 2
        A2_phi = A1_phi
    
        initial_guess_stage1 = [0, 2, 0, 2, 0, 2, 0, 2, theor_freqs_GHz[0], theor_freqs_GHz[1]]
        lower_bounds_stage1 = [-np.pi, 0.01, -np.pi, 0.01, -np.pi, 0.01, -np.pi, 0.01, 0.1, 0.1]
        upper_bounds_stage1 = [np.pi, 100, np.pi, 100, np.pi, 100, np.pi, 100, 120, 120]
    
        result_stage1 = least_squares(
            residuals_stage1,
            x0=initial_guess_stage1,
            bounds=(lower_bounds_stage1, upper_bounds_stage1),
            args=(sim_time, theta, phi, A1_theta, A2_theta, A1_phi, A2_phi),
            xtol=1e-4, ftol=1e-4, gtol=1e-4, max_nfev=10000
        )
        (f1_theta_opt, n1_theta_opt, f2_theta_opt, n2_theta_opt,
         f1_phi_opt, n1_phi_opt, f2_phi_opt, n2_phi_opt,
         f1_GHz_opt, f2_GHz_opt) = result_stage1.x
    
        initial_guess_stage2 = [A1_theta, A2_theta, A1_phi, A2_phi]
        result_stage2 = least_squares(
            residuals_stage2,
            x0=initial_guess_stage2,
            args=(sim_time, theta, phi, f1_theta_opt, n1_theta_opt, f2_theta_opt, n2_theta_opt,
                  f1_phi_opt, n1_phi_opt, f2_phi_opt, n2_phi_opt, f1_GHz_opt, f2_GHz_opt),
            xtol=1e-4, ftol=1e-4, gtol=1e-4, max_nfev=10000
        )
        A1_theta_opt, A2_theta_opt, A1_phi_opt, A2_phi_opt = result_stage2.x
    
        initial_guess_stage3 = [
            A1_theta_opt, f1_theta_opt, n1_theta_opt, A2_theta_opt, f2_theta_opt, n2_theta_opt,
            A1_phi_opt, f1_phi_opt, n1_phi_opt, A2_phi_opt, f2_phi_opt, n2_phi_opt,
            f1_GHz_opt, f2_GHz_opt
        ]
        lower_bounds_stage3 = [
            -np.inf, -np.pi, 0.01, -np.inf, -np.pi, 0.01,
            -np.inf, -np.pi, 0.01, -np.inf, -np.pi, 0.01, 0.1, 0.1
        ]
        upper_bounds_stage3 = [
            np.inf, np.pi, 100, np.inf, np.pi, 100,
            np.inf, np.pi, 100, np.inf, np.pi, 100, 120, 120
        ]
        result_stage3 = least_squares(
            combined_residuals,
            x0=initial_guess_stage3,
            bounds=(lower_bounds_stage3, upper_bounds_stage3),
            args=(sim_time, theta, phi),
            xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=10000
        )
        opt_params = result_stage3.x
        (A1_theta_opt, f1_theta_opt, n1_theta_opt, A2_theta_opt, f2_theta_opt, n2_theta_opt,
         A1_phi_opt, f1_phi_opt, n1_phi_opt, A2_phi_opt, f2_phi_opt, n2_phi_opt,
         f1_GHz_opt, f2_GHz_opt) = opt_params
    
        theta_fit = fit_function_theta(sim_time, A1_theta_opt, f1_theta_opt, n1_theta_opt,
                                   A2_theta_opt, f2_theta_opt, n2_theta_opt,
                                   f1_GHz_opt, f2_GHz_opt)
        phi_fit = fit_function_phi(sim_time, A1_phi_opt, f1_phi_opt, n1_phi_opt,
                                   A2_phi_opt, f2_phi_opt, n2_phi_opt,
                                   f1_GHz_opt, f2_GHz_opt)
        
        approx_freqs_GHz = sorted(np.round([f1_GHz_opt, f2_GHz_opt], 1), reverse=True)

        phi_fig = create_phi_fig(time_ns, phi, phi_fit, H, T, approx_freqs_GHz, theor_freqs_GHz, material)
        theta_fig = create_theta_fig(time_ns, theta, theta_fit)

    else:
        approx_freqs_GHz = (None, None)
        theta_fit = None
        phi_fit = None
    
    # Далее строим графики
    phi_fig = create_phi_fig(time_ns, phi, phi_fit, H, T, approx_freqs_GHz, theor_freqs_GHz, material)
    theta_fig = create_theta_fig(time_ns, theta, theta_fit)
    yz_fig = create_yz_fig(np.sin(np.pi/2 + np.radians(theta)) * np.sin(np.radians(phi)),
                           np.cos(np.pi/2 + np.radians(theta)),
                           time_ns)
    if material_changed:
        phi_amp_fig = create_phi_amp_fig(T_vals, H_vals, amplitude_phi_static)
        theta_amp_fig = create_theta_amp_fig(T_vals, H_vals, amplitude_theta_static)
        freq_fig = create_freq_fig(T_vals, H_vals, freq_array1, freq_array2)
    elif params_changed or switch_on:
        phi_amp_fig = no_update
        theta_amp_fig = no_update
        freq_fig = create_freq_fig(T_vals, H_vals, freq_array1, freq_array2)
    else:
        phi_amp_fig = no_update
        theta_amp_fig = no_update
        freq_fig = no_update

    return phi_fig, theta_fig, yz_fig, phi_amp_fig, theta_amp_fig, freq_fig

@app.callback(
    Output('download-H-file', 'data'),
    Input('download-H-btn',  'n_clicks'),
    State('H-slider',        'value'),
    State('param-store',       'data'),
    State('material-dropdown', 'value'),
    prevent_initial_call=True,
)
def download_hfix(n_clicks, H, store, material):
    if not n_clicks:
        raise PreventUpdate
    def _make_hfix_npy(H, p, material):
        """Возвращает bytes содержимого .npy для фиксированного H."""
        chi_scale = 10 ** p.chi_scale
        k_scale   = 10 ** p.k_scale
        m_scale   = 10 ** p.m_scale
        T_vals  = T_vals_1 if material == '1' else T_vals_2
        m_vec   = m_scale * (m_array_1 if material == '1' else m_array_2)
        chi_vec = chi_scale * (chi_array_1 if material == '1' else chi_array_2)
        K_vec   = k_scale * (K_array_1 if material == '1' else K_array_2)
    
        f1, f2 = compute_frequencies_H_fix(H, m_vec, chi_vec, K_vec, gamma)
        arr = np.vstack([T_vals, f1, f2])           # shape (3, N)
    
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        return buf.getvalue()
    p = SimParams(**store[material])
    content = _make_hfix_npy(H, p, material)
    return dcc.send_bytes(content, filename=f'H_{H/10:.0f}.npy')


@app.callback(
    Output('download-T-file', 'data'),
    Input('download-T-btn',   'n_clicks'),
    State('T-slider',         'value'),
    State('param-store',       'data'),
    State('material-dropdown', 'value'),
    prevent_initial_call=True,
)
def download_tfix(n_clicks, T, store, material):
    if not n_clicks:
        raise PreventUpdate
    def _make_tfix_npy(T, p, material):
        """Возвращает bytes содержимого .npy для фиксированной T."""
        chi_scale = 10 ** p.chi_scale
        k_scale   = 10 ** p.k_scale
        m_scale   = 10 ** p.m_scale
        H_vec = H_vals
        # индексы и скаляры при выбранной температуре
        t_idx   = np.abs((T_vals_1 if material == '1' else T_vals_2) - T).argmin()
        m_val   = m_scale * (m_array_1 if material == '1' else m_array_2)[t_idx]
        chi_val = chi_scale * (chi_array_1 if material == '1' else chi_array_2)[t_idx]
        K_val   = k_scale * (K_array_1  if material == '1' else K_array_2)[t_idx]
    
        f1, f2 = compute_frequencies_T_fix(H_vec, m_val, chi_val, K_val, gamma)
        arr = np.vstack([H_vec, f1, f2])            # shape (3, N)
    
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        return buf.getvalue()
    p = SimParams(**store[material])
    content = _make_tfix_npy(T, p, material)
    return dcc.send_bytes(content, filename=f'T_{T:.0f}.npy')

if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0", port=8000)
