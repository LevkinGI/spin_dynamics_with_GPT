titles = ('H = 100 мТл', 'T = 293 K', 'T = 310 K', 'T = 323 K')
fig = make_subplots(
    rows=2, cols=4,
    vertical_spacing=0.04,
    horizontal_spacing=0.04,
    shared_xaxes=True,
    shared_yaxes=True,
    subplot_titles=titles
)

for c, var in enumerate(['H_100', 'T_293', 'T_310', 'T_323'], start=1):
    if var == 'H_100':
        axis, hf, lf, hf_tau, lf_tau = _load_theor_curves(Path(data_dir), var)
        exp_axis, exp_hf, exp_lf, exp_hf_tau, exp_lf_tau = H_100_axis, H_100_hf, H_100_lf, H_100_hf_tau, H_100_lf_tau
        varying, var_label, val = ("H", "мТл", 100)
        axis_title = 'Температура (K)'
    elif var == 'T_293':
        axis, hf, lf, hf_tau, lf_tau = _load_theor_curves(Path(data_dir), var)
        exp_axis, exp_hf, exp_lf, exp_hf_tau, exp_lf_tau = T_293_axis, T_293_hf, T_293_lf, T_293_hf_tau, T_293_lf_tau
        varying, var_label, val = ("T", "K", 293)
        axis_title = 'Магнитное поле (мТл)'
    elif var == 'T_310':
        axis, hf, lf, hf_tau, lf_tau = _load_theor_curves(Path(data_dir), var)
        exp_axis, exp_hf, exp_lf, exp_hf_tau, exp_lf_tau = T_310_axis, T_310_hf, T_310_lf, T_310_hf_tau, T_310_lf_tau
        varying, var_label, val = ("T", "K", 310)
        axis_title = 'Магнитное поле (мТл)'
    elif var == 'T_323':
        axis, hf, lf, hf_tau, lf_tau = _load_theor_curves(Path(data_dir), var)
        exp_axis, exp_hf, exp_lf, exp_hf_tau, exp_lf_tau = T_323_axis, T_323_hf, T_323_lf, T_323_hf_tau, T_323_lf_tau
        varying, var_label, val = ("T", "K", 323)
        axis_title = 'Магнитное поле (мТл)'
    
    fig.add_trace(go.Scatter(x=axis, y=lf, mode="lines",
                             line=dict(width=2, color=FIT_LF),
                             name=f"f_LF, {varying} = {val} {var_label}"), row=1, col=c)
    fig.add_trace(go.Scatter(x=exp_axis, y=exp_lf, mode="markers",
                             line=dict(width=2, color=FIT_LF),
                             marker=dict(size=9, color=FIT_LF),
                             name=f"f_LF, {varying} = {val} {var_label}"), row=1, col=c)
    fig.add_trace(go.Scatter(x=axis, y=hf, mode="lines",
                             line=dict(width=2, color=FIT_HF),
                             name=f"f_HF, {varying} = {val} {var_label}"), row=1, col=c)
    fig.add_trace(go.Scatter(x=exp_axis, y=exp_hf, mode="markers",
                             line=dict(width=2, color=FIT_HF),
                             marker=dict(size=9, color=FIT_HF),
                             name=f"f_LF, {varying} = {val} {var_label}"), row=1, col=c)
    
    fig.add_trace(go.Scatter(x=axis, y=lf_tau, mode="lines",
                             line=dict(width=2, color=FIT_LF),
                             name=f"tau_LF, {varying} = {val} {var_label}"), row=2, col=c)
    fig.add_trace(go.Scatter(x=exp_axis, y=exp_lf_tau, mode="markers",
                             line=dict(width=2, color=FIT_LF),
                             marker=dict(size=9, color=FIT_LF),
                             name=f"f_LF, {varying} = {val} {var_label}"), row=2, col=c)
    fig.add_trace(go.Scatter(x=axis, y=hf_tau, mode="lines",
                             line=dict(width=2, color=FIT_HF),
                             name=f"tau_HF, {varying} = {val} {var_label}"), row=2, col=c)
    fig.add_trace(go.Scatter(x=exp_axis, y=exp_hf_tau, mode="markers",
                             line=dict(width=2, color=FIT_HF),
                             marker=dict(size=9, color=FIT_HF),
                             name=f"f_LF, {varying} = {val} {var_label}"), row=2, col=c)
    
    # fig.update_yaxes(range=[0, 0.4], row=2, col=c)
    fig.update_xaxes(title_text=axis_title, row=2, col=c)
    for r in (1, 2):
        fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True,
                         showgrid=True, gridcolor="#cccccc", gridwidth=1, row=r, col=c)
        fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True,
                         showgrid=True, gridcolor="#cccccc", gridwidth=1, row=r, col=c)


fig.update_yaxes(title_text="Частота (ГГц)", row=1, col=1)
fig.update_yaxes(title_text="Время затухания (нс)", row=2, col=1)
fig.update_layout(
    showlegend=False, hovermode="x unified",
    font=dict(size=16), width=1900, height=800,
    paper_bgcolor="white", plot_bgcolor="white",
)
for annotation in fig.layout.annotations:
    annotation.font = dict(size=28)
fig.update_xaxes(title_font=dict(size=28), tickfont=dict(size=24))
fig.update_yaxes(title_font=dict(size=28), tickfont=dict(size=24))
