import hashlib
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy.spatial import distance_matrix

from teaching_math import (
    apply_zone_edits,
    expected_origin_flows,
    fast_playback_frames,
    flow_highlight_payload,
    playback_frame_snapshot,
    playback_neighbor,
    poisson_sample,
    prune_zone_edits,
    run_ipf,
    scale_destination_targets,
    select_display_matrix,
    selected_od_snapshot,
    teaching_destination_targets,
    update_zone_edit,
)


# ------------------------------
# Streamlit App Header (original)
# ------------------------------
st.set_page_config(page_title="LUTI Classroom Demo", layout="wide")
st.title("Land-Use — Transport Interaction (LUTI) Model")
st.markdown("A simple simulation of land-use and transport interaction.")


# ------------------------------
# Helper functions
# ------------------------------
def decay_exp(d, beta):
    return np.exp(-beta * d)


def fit_distance_decay(T, Oi, Aj, dist):
    """Original fitting routine, isolated so a failed fit does not affect simulation beta."""
    rows, cols = np.where(T > 0)
    if len(rows) == 0:
        return {'k': np.nan, 'beta': np.nan, 'beta_se': np.nan}
    y = T[rows, cols].astype(float)
    dvec = dist[rows, cols].astype(float)
    denom = np.maximum(Oi[rows] * Aj[cols], 1e-6)
    target = y / denom
    valid = np.isfinite(target) & (target > 0) & np.isfinite(dvec)
    if valid.sum() < 10:
        return {'k': np.nan, 'beta': np.nan, 'beta_se': np.nan}
    try:
        popt, pcov = curve_fit(
            lambda d, k, beta: k * np.exp(-beta * d),
            dvec[valid],
            target[valid],
            p0=[0.001, 0.05],
            maxfev=20000,
        )
        perr = np.sqrt(np.diag(pcov))
        return {
            'k': popt[0],
            'beta': popt[1],
            'beta_se': perr[1] if len(perr) > 1 else np.nan,
        }
    except Exception:
        return {'k': np.nan, 'beta': np.nan, 'beta_se': np.nan}


def plot_zones(zones):
    fig = px.scatter(
        zones,
        x='x',
        y='y',
        size='pop',
        color='attractiveness',
        hover_data=['zone', 'pop', 'jobs', 'accessibility', 'attractiveness'],
        color_continuous_scale='Viridis',
        title="Zones (size=population, color=attractiveness)",
    )
    fig.update_layout(width=600, height=400)
    return fig


def plot_decay(true_beta, fitted_results, modes):
    dvals = np.linspace(0, 140, 400)
    fig = make_subplots(
        rows=1,
        cols=len(modes),
        subplot_titles=[f"{m} decay fitting" for m in modes],
        shared_yaxes=True,
    )
    for i, m in enumerate(modes, start=1):
        y_true = decay_exp(dvals, true_beta[m])
        fig.add_trace(
            go.Scatter(
                x=dvals,
                y=y_true,
                mode='lines',
                name=f"Theoretical {m} (β={true_beta[m]:.3f})",
                line=dict(width=2, color='royalblue'),
                legendgroup=m,
                showlegend=True,
            ),
            row=1,
            col=i,
        )
        if not np.isnan(fitted_results[m]['beta']):
            y_fit = fitted_results[m]['k'] * np.exp(-fitted_results[m]['beta'] * dvals)
            fig.add_trace(
                go.Scatter(
                    x=dvals,
                    y=y_fit,
                    mode='lines',
                    name=f"Fitted {m} (β={fitted_results[m]['beta']:.3f})",
                    line=dict(width=2, dash='dash', color='crimson'),
                    legendgroup=m,
                    showlegend=True,
                ),
                row=1,
                col=i,
            )
        with np.errstate(divide='ignore', invalid='ignore'):
            long_tail = dvals ** -1.5
        long_tail[~np.isfinite(long_tail)] = np.nan
        fig.add_trace(
            go.Scatter(
                x=dvals,
                y=long_tail / np.nanmax(long_tail),
                mode='lines',
                name='Power-law tail ∝ d^-1.5',
                line=dict(dash='dot', color='gray'),
                legendgroup=m,
                showlegend=True,
            ),
            row=1,
            col=i,
        )
        fig.update_xaxes(title_text="Distance", row=1, col=i)
        if i == 1:
            fig.update_yaxes(title_text="Decay Function", type='linear', row=1, col=i)
    fig.update_layout(
        title="Distance Decay Fitting (True vs Fitted)",
        yaxis_type='linear',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1,
            traceorder="grouped",
            itemsizing='constant',
        ),
        margin=dict(b=120),
    )
    return fig


def plot_flows(
    T, mode_sel, constraint_sel, percentile, coords_mat, zones,
    highlighted_od=None, matrix_label=None,
):
    """Original Plotly flow map, now fed the single matrix used everywhere else."""
    vals = T[np.isfinite(T) & (T > 0)]
    thresh = np.percentile(vals, percentile) if vals.size > 0 else 0
    visible = T[np.isfinite(T) & (T > thresh)]
    if len(visible) == 0:
        visible = np.array([1.0])
    max_flow, min_flow = np.nanmax(visible), np.nanmin(visible)
    fig = go.Figure()
    for i in range(len(T)):
        for j in range(len(T)):
            if i != j and np.isfinite(T[i, j]) and T[i, j] >= thresh and T[i, j] > 0:
                normalized = (T[i, j] - min_flow) / (max_flow - min_flow + 1e-9)
                normalized = np.clip(normalized, 0, 1)
                width = 1 + 8 * (normalized ** 0.25)
                if not np.isfinite(width):
                    width = 1.0
                fig.add_trace(
                    go.Scatter(
                        x=[coords_mat[i, 0], coords_mat[j, 0]],
                        y=[coords_mat[i, 1], coords_mat[j, 1]],
                        mode='lines',
                        line=dict(color='crimson', width=float(width)),
                        hoverinfo='none',
                        showlegend=False,
                    )
                )
    if highlighted_od is not None:
        selected_i, selected_j = highlighted_od
        highlight = flow_highlight_payload(T, coords_mat, selected_i, selected_j)
        fig.add_trace(
            go.Scatter(
                x=highlight['x'],
                y=highlight['y'],
                mode='lines+markers',
                line=dict(color='gold', width=7),
                marker=dict(size=10, color=['darkorange', 'gold'], line=dict(width=1, color='black')),
                name=f'Selected {selected_i} → {selected_j}',
                customdata=[highlight['flow'], highlight['flow']],
                hovertemplate=(
                    f'Selected OD {selected_i} → {selected_j}<br>'
                    'Flow: %{customdata:,.6f}<extra></extra>'
                ),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=zones['x'],
            y=zones['y'],
            mode='markers+text',
            text=zones['zone'],
            marker=dict(size=zones['pop'] * 0.0006, color='royalblue', opacity=0.7),
            textposition='top center',
            name='zones',
        )
    )
    fig.update_layout(
        title=(
            f"Top {100 - percentile}% flows ({mode_sel}, {constraint_sel})"
            + (f"<br><sup>{matrix_label}</sup>" if matrix_label else "")
        ),
        width=600,
        height=350,
        xaxis=dict(
            range=[0, 100], showgrid=False, zeroline=False, mirror=True,
            linecolor='grey', linewidth=1, ticks='', showticklabels=False,
        ),
        yaxis=dict(
            range=[0, 100], showgrid=False, zeroline=False, mirror=True,
            linecolor='black', linewidth=1, ticks='', showticklabels=False,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def matrix_with_totals(matrix):
    labels = [str(i) for i in range(matrix.shape[0])]
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame['Σ row'] = matrix.sum(axis=1)
    frame.loc['Σ col'] = list(matrix.sum(axis=0)) + [matrix.sum()]
    return frame


# ------------------------------
# UI controls for experiment (original controls retained)
# ------------------------------
np.random.seed(2025)
N = st.sidebar.slider("Number of zones", 20, 60, 40)

modes = ['walk_transit', 'car', 'long_distance_bus']
default_betas = {'walk_transit': 0.12, 'car': 0.04, 'long_distance_bus': 0.02}
mode_trip_rate = {'walk_transit': 0.25, 'car': 0.45, 'long_distance_bus': 0.10}
# Retained for source compatibility and audit visibility.  It is not multiplied
# into the destination weights because each mode-wide scalar cancels exactly
# during row normalisation and therefore has no effect on probabilities.
mode_job_pref = {'walk_transit': 1.2, 'car': 1.0, 'long_distance_bus': 0.8}

st.sidebar.header("Flow display")
mode_sel = st.sidebar.selectbox("Select transport mode:", modes)
constraint_sel = st.sidebar.radio(
    "Constraint type:", ["Origin-Constrained", "Doubly-Constrained"]
)
percentile = st.sidebar.slider("Show strongest flows (percentile):", 80, 99, 97)

st.sidebar.header("Distance-decay β (adjustable)")
beta_sliders = {}
for mode in modes:
    beta_sliders[mode] = st.sidebar.slider(
        f"{mode} β", 0.001, 0.5, float(default_betas[mode]), step=0.001
    )

st.sidebar.header("Teaching calculation")
use_poisson = st.sidebar.checkbox(
    "Use Poisson random simulation",
    value=False,
    help="Off by default so expected OD flows preserve every origin total.",
)
sample_seed = st.sidebar.number_input(
    "Poisson seed", min_value=0, max_value=1_000_000, value=2025, step=1,
    disabled=not use_poisson,
)
show_balance_process = False
rescale_sampled_destinations = False
destination_target_source = "Original"
if constraint_sel == "Doubly-Constrained":
    show_balance_process = st.sidebar.checkbox("Show balancing process", value=False)
    destination_target_source = st.sidebar.radio(
        "Destination target source:",
        ["Teaching Scenario", "Original"],
        index=0,
        help=(
            "Teaching Scenario sets Dⱼ from each zone's share of jobs. "
            "Original preserves the prior behavior and uses current OD column totals."
        ),
    )
    rescale_sampled_destinations = st.sidebar.checkbox(
        "Explicitly scale D targets if totals differ",
        value=False,
        help="Applies D′j = Dj × ΣO / ΣD. The scale factor is always reported.",
        disabled=destination_target_source != "Original",
    )


# ------------------------------
# Generate original synthetic city
# ------------------------------
coords = np.random.uniform(0, 100, size=(N, 2))
pop = np.random.lognormal(mean=8.5, sigma=0.6, size=N).astype(int)
jobs = (pop * np.random.uniform(0.3, 1.2, size=N)).astype(int)
zones = pd.DataFrame({
    'zone': np.arange(N),
    'x': coords[:, 0],
    'y': coords[:, 1],
    'pop': pop,
    'jobs': jobs,
})
base_population = zones['pop'].to_numpy(copy=True)
base_jobs = zones['jobs'].to_numpy(copy=True)
if "luti_zone_edits" not in st.session_state:
    st.session_state.luti_zone_edits = {}
if st.session_state.get("luti_zone_count") != N:
    for state_key in list(st.session_state.keys()):
        if state_key.startswith("teach_population_") or state_key.startswith("teach_jobs_"):
            del st.session_state[state_key]
    st.session_state.luti_zone_count = N
st.session_state.luti_zone_edits = prune_zone_edits(
    st.session_state.luti_zone_edits, N
)
effective_population, effective_jobs = apply_zone_edits(
    base_population, base_jobs, st.session_state.luti_zone_edits
)
zones['pop'] = effective_population
zones['jobs'] = effective_jobs


# ------------------------------
# Teaching controls: native Streamlit, placed above the original plots
# ------------------------------
with st.expander("🧮 Teaching Mode / Step-by-Step Calculation", expanded=False):
    teaching_enabled = st.checkbox(
        "Enable teaching controls",
        value=False,
        help="Enables one-zone editing and numerical substitution below.",
    )
    if st.button("Restore initial city data", use_container_width=False):
        st.session_state.luti_zone_edits = {}
        for state_key in list(st.session_state.keys()):
            if state_key.startswith("teach_population_") or state_key.startswith("teach_jobs_"):
                del st.session_state[state_key]
        rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun")
        rerun()
    st.caption(
        f"Saved edits: {len(st.session_state.luti_zone_edits)} zone(s). "
        "Edits persist across mode, β and constraint changes."
    )
    selector_cols = st.columns(3)
    with selector_cols[0]:
        origin_i = st.selectbox(
            "Origin i", range(N), index=0, format_func=lambda value: f"Zone {value}",
            key=f"teach_origin_{N}",
        )
    with selector_cols[1]:
        destination_j = st.selectbox(
            "Destination j", range(N), index=min(1, N - 1),
            format_func=lambda value: f"Zone {value}",
            key=f"teach_destination_{N}",
        )
    with selector_cols[2]:
        edit_zone = st.selectbox(
            "Zone to edit", range(N), index=origin_i,
            format_func=lambda value: f"Zone {value}",
            disabled=not teaching_enabled,
            key=f"teach_edit_zone_{N}",
        )

    edit_cols = st.columns(2)
    with edit_cols[0]:
        edited_population = st.number_input(
            f"Zone {edit_zone} population",
            min_value=0,
            max_value=1_000_000,
            value=int(zones.loc[edit_zone, 'pop']),
            step=100,
            key=f"teach_population_{N}_{edit_zone}",
            disabled=not teaching_enabled,
        )
    with edit_cols[1]:
        edited_jobs = st.number_input(
            f"Zone {edit_zone} jobs",
            min_value=0,
            max_value=1_000_000,
            value=int(zones.loc[edit_zone, 'jobs']),
            step=100,
            key=f"teach_jobs_{N}_{edit_zone}",
            disabled=not teaching_enabled,
        )

    step_labels = [
        "1 · Trip generation",
        "2 · Destination attractiveness",
        "3 · Distance decay",
        "4 · Weight",
        "5 · Probability",
        "6 · Predicted OD flow",
    ]
    selected_step = st.radio(
        "Calculation step",
        step_labels,
        horizontal=True,
        disabled=not teaching_enabled,
    )
    show_all_steps = st.checkbox(
        "Show all steps together", value=False, disabled=not teaching_enabled
    )
    teaching_output = st.container()

if teaching_enabled:
    st.session_state.luti_zone_edits = update_zone_edit(
        st.session_state.luti_zone_edits,
        edit_zone,
        edited_population,
        edited_jobs,
        N,
    )
    effective_population, effective_jobs = apply_zone_edits(
        base_population, base_jobs, st.session_state.luti_zone_edits
    )
    zones['pop'] = effective_population
    zones['jobs'] = effective_jobs


# ------------------------------
# Accessibility and Attractiveness (original formulation)
# ------------------------------
coords_mat = zones[['x', 'y']].values
dist = distance_matrix(coords_mat, coords_mat)
np.fill_diagonal(dist, 0.001)

beta_acc = 0.03
accessibility_raw = decay_exp(dist, beta_acc).dot(zones['jobs'].values)
accessibility = (
    (accessibility_raw - accessibility_raw.min())
    / (accessibility_raw.max() - accessibility_raw.min() + 1e-9)
)
alpha_jobs = 0.7
alpha_acc = 0.3
attractiveness_raw = (
    (zones['jobs'].values ** alpha_jobs) * ((1 + accessibility) ** alpha_acc)
)
mean_attractiveness = np.mean(attractiveness_raw)
if mean_attractiveness > 0:
    attractiveness = attractiveness_raw / mean_attractiveness
else:
    attractiveness = np.zeros_like(attractiveness_raw, dtype=float)
zones['accessibility'] = accessibility
zones['attractiveness'] = attractiveness


# ------------------------------
# Mode-level supplies and unified OD matrices
# ------------------------------
O_mode = {mode: zones['pop'].values * mode_trip_rate[mode] for mode in modes}
expected_flows_mode = {}
probabilities_mode = {}
weights_mode = {}
flows_mode = {}
calculation_error = None

try:
    for mode_index, mode in enumerate(modes):
        expected, probabilities, weights = expected_origin_flows(
            O_mode[mode], attractiveness, dist, beta_sliders[mode]
        )
        expected_flows_mode[mode] = expected
        probabilities_mode[mode] = probabilities
        weights_mode[mode] = weights
        flows_mode[mode] = (
            poisson_sample(expected, int(sample_seed) + mode_index)
            if use_poisson else expected.copy()
        )
except ValueError as error:
    calculation_error = str(error)
    for mode in modes:
        expected_flows_mode[mode] = np.zeros((N, N), dtype=float)
        probabilities_mode[mode] = np.zeros((N, N), dtype=float)
        weights_mode[mode] = np.zeros((N, N), dtype=float)
        flows_mode[mode] = np.zeros((N, N), dtype=float)


# ------------------------------
# Fit distance decay (diagnostic only; never replaces slider beta)
# ------------------------------
fitted_results = {
    mode: fit_distance_decay(
        flows_mode[mode], O_mode[mode], attractiveness, dist
    )
    for mode in modes
}


# ------------------------------
# Select the one matrix used by Flow Map, teaching values and OD display
# ------------------------------
origin_matrix = flows_mode[mode_sel].copy()
display_matrix = origin_matrix.copy()
display_matrix_label = "Origin-Constrained"
display_matrix_valid = True
display_value_unit = "trips"
playback_signature = None
ipf_result = None
destination_targets = display_matrix.sum(axis=0)
destination_scale_factor = 1.0
origin_targets = O_mode[mode_sel].astype(float)
base_matrix = attractiveness[np.newaxis, :] * decay_exp(dist, beta_sliders[mode_sel])

if constraint_sel == "Doubly-Constrained":
    if destination_target_source == "Teaching Scenario":
        destination_targets = teaching_destination_targets(
            origin_targets, zones['jobs'].values
        )
    else:
        destination_targets = origin_matrix.sum(axis=0).astype(float)
    total_gap = abs(origin_targets.sum() - destination_targets.sum())
    if (
        destination_target_source == "Original"
        and total_gap > 1e-6
        and rescale_sampled_destinations
    ):
        destination_targets, destination_scale_factor = scale_destination_targets(
            origin_targets, destination_targets
        )
    ipf_result = run_ipf(
        origin_targets,
        destination_targets,
        base_matrix,
        max_iter=500,
        tol=1e-6,
    )
    display_matrix, display_matrix_label, display_matrix_valid = select_display_matrix(
        origin_matrix, ipf_result
    )

    if ipf_result.converged and show_balance_process:
        signature = hashlib.sha1(
            origin_targets.tobytes()
            + destination_targets.tobytes()
            + base_matrix.tobytes()
        ).hexdigest()
        playback_signature = signature
        if st.session_state.get("ipf_signature") != signature:
            st.session_state.ipf_signature = signature
            st.session_state.ipf_frame = 0
            st.session_state.ipf_autoplay = False
        if "ipf_frame" not in st.session_state:
            st.session_state.ipf_frame = 0
        if "ipf_autoplay" not in st.session_state:
            st.session_state.ipf_autoplay = False
        st.session_state.ipf_frame = min(
            int(st.session_state.ipf_frame), len(ipf_result.history) - 1
        )
        display_matrix, display_matrix_label, display_matrix_valid = select_display_matrix(
            origin_matrix, ipf_result, st.session_state.ipf_frame
        )
        active_frame = playback_frame_snapshot(
            ipf_result.history, st.session_state.ipf_frame
        )
        display_value_unit = active_frame['unit']

selected_snapshot = selected_od_snapshot(display_matrix, origin_i, destination_j)


# ------------------------------
# Fill the teaching expander with live formula substitution
# ------------------------------
def render_teaching_step(step_number):
    i = int(origin_i)
    j = int(destination_j)
    mode = mode_sel
    beta = beta_sliders[mode]
    Oi = O_mode[mode][i]
    Aj = attractiveness[j]
    dij = dist[i, j]
    decay = decay_exp(dij, beta)
    weight = weights_mode[mode][i, j]
    denominator = weights_mode[mode][i, :].sum()
    probability = probabilities_mode[mode][i, j]
    expected_flow = expected_flows_mode[mode][i, j]
    snapshot = selected_od_snapshot(display_matrix, i, j)
    used_flow = snapshot['flow']

    if step_number == 1:
        st.markdown("#### Step 1 · Population and trip generation")
        st.latex(r"O_i = Pop_i \times r_i")
        st.code(
            f"O_{i} = {zones.loc[i, 'pop']:,} × {mode_trip_rate[mode]:.2f} "
            f"= {Oi:,.3f} trips"
        )
    elif step_number == 2:
        st.markdown("#### Step 2 · Destination accessibility and attractiveness")
        st.latex(r"Acc_j^{raw}=\sum_m Jobs_m e^{-\beta_{acc}d_{jm}}")
        st.latex(r"A_j^{raw}=Jobs_j^{0.7}(1+Acc_j)^{0.3},\quad A_j=A_j^{raw}/\overline{A^{raw}}")
        st.code(
            f"Zone {j}: Jobs={zones.loc[j, 'jobs']:,}; "
            f"Acc_raw={accessibility_raw[j]:,.4f}; Acc={accessibility[j]:.6f}; "
            f"A_raw={attractiveness_raw[j]:,.4f}; A={Aj:.6f}"
        )
        stat_cols = st.columns(4)
        stat_cols[0].metric("Minimum Acc_raw", f"{accessibility_raw.min():,.3f}")
        stat_cols[1].metric("Maximum Acc_raw", f"{accessibility_raw.max():,.3f}")
        stat_cols[2].metric("Mean A_raw", f"{mean_attractiveness:,.3f}")
        stat_cols[3].metric("Mean normalised A", f"{attractiveness.mean():.6f}")
        st.code(
            f"Acc_{j} = ({accessibility_raw[j]:,.4f} − {accessibility_raw.min():,.4f}) "
            f"/ ({accessibility_raw.max():,.4f} − {accessibility_raw.min():,.4f} + 1e−9) "
            f"= {accessibility[j]:.6f}\n"
            f"A_{j} = {attractiveness_raw[j]:,.4f} / {mean_attractiveness:,.4f} "
            f"= {Aj:.6f}"
        )
        st.caption("β_acc = 0.03, α_jobs = 0.7 and α_acc = 0.3 are the original example settings.")
    elif step_number == 3:
        st.markdown("#### Step 3 · Distance decay")
        st.latex(r"f(d_{ij})=e^{-\beta d_{ij}}")
        st.code(f"exp(−{beta:.3f} × {dij:.4f}) = {decay:.8f}")
    elif step_number == 4:
        st.markdown("#### Step 4 · Unnormalised destination weight")
        st.latex(r"w_{ij}=A_j e^{-\beta d_{ij}}")
        st.code(f"w_{i},{j} = {Aj:.6f} × {decay:.8f} = {weight:.8f}")
        st.info(
            "The original mode_job_pref was a mode-wide scalar. It cancelled during "
            "row normalisation, so removing it does not change any destination probability."
        )
    elif step_number == 5:
        st.markdown("#### Step 5 · Normalised destination probability")
        st.latex(r"p_{ij}=\frac{w_{ij}}{\sum_m w_{im}}")
        st.code(
            f"p_{i},{j} = {weight:.8f} / {denominator:.8f} "
            f"= {probability:.8f} ({probability * 100:.4f}%)"
        )
        st.caption(f"Probability sum for origin {i}: {probabilities_mode[mode][i, :].sum():.12f}")
        contributions = pd.DataFrame({
            'destination': np.arange(N),
            'attractiveness A': attractiveness,
            'decay f(d)': decay_exp(dist[i, :], beta),
            'weight w': weights_mode[mode][i, :],
            'probability p': probabilities_mode[mode][i, :],
        }).sort_values('weight w', ascending=False)
        st.markdown(
            f"**Denominator composition:** all {N} destination weights sum to "
            f"`{denominator:.8f}`. The largest contributors are shown below."
        )
        st.dataframe(
            contributions.head(8).style.format({
                'attractiveness A': '{:.6f}',
                'decay f(d)': '{:.8f}',
                'weight w': '{:.8f}',
                'probability p': '{:.4%}',
            }),
            use_container_width=True,
            height=250,
        )
        with st.expander(f"All {N} denominator contributions"):
            st.dataframe(contributions, use_container_width=True, height=320)
    elif step_number == 6:
        st.markdown("#### Step 6 · Predicted OD flow")
        st.latex(r"E[T_{ij}]=O_i p_{ij}")
        st.code(
            f"E[T_{i},{j}] = {Oi:,.3f} × {probability:.8f} "
            f"= {expected_flow:,.6f} trips"
        )
        if use_poisson and constraint_sel == "Origin-Constrained":
            st.warning(
                f"Poisson simulation is enabled. The matrix and Flow Map use the sampled "
                f"value {used_flow:,.0f}, not the expectation {expected_flow:,.6f}."
            )
        elif constraint_sel == "Doubly-Constrained" and ipf_result and ipf_result.converged:
            st.info(
                f"The matrix currently shown on the Flow Map is **{display_matrix_label}**; "
                f"this cell is {used_flow:,.6f}."
            )
        elif constraint_sel == "Doubly-Constrained" and not display_matrix_valid:
            st.error(
                "IPF did not converge. This value and the Flow Map use the valid "
                "Origin-Constrained fallback, not a zero matrix or a claimed double constraint."
            )
        st.metric(
            f"Selected OD {i} → {j}",
            f"{used_flow:,.6f} {display_value_unit}",
        )
        with st.expander("OD matrix currently used by the Flow Map"):
            st.dataframe(
                matrix_with_totals(display_matrix).style.format("{:.3f}"),
                use_container_width=True,
                height=320,
            )


with teaching_output:
    if teaching_enabled:
        if calculation_error:
            st.error(calculation_error)
        else:
            if show_all_steps:
                for step_number in range(1, 7):
                    render_teaching_step(step_number)
                    if step_number < 6:
                        st.divider()
            else:
                render_teaching_step(step_labels.index(selected_step) + 1)
    else:
        st.caption("Enable teaching controls to edit one zone and inspect live calculations.")


# ------------------------------
# Original layout and plots
# ------------------------------
if calculation_error:
    st.error(f"OD calculation stopped: {calculation_error}")
if ipf_result is not None:
    if destination_target_source == "Teaching Scenario":
        st.info(
            "Teaching Scenario destination targets use "
            "Dⱼ = ΣO × Jobsⱼ / ΣJobs. This is a new classroom assumption for "
            "demonstrating balancing, not an empirical calibration result of the original model."
        )
    if ipf_result.status == "total_mismatch":
        st.error(
            f"Doubly-Constrained model not run: {ipf_result.message} "
            "Enable the explicit D-target scaling option or turn off Poisson simulation."
        )
    elif ipf_result.status == "infeasible":
        st.error(f"Doubly-Constrained model infeasible: {ipf_result.message}")
    elif ipf_result.status == "max_iterations":
        st.error(ipf_result.message)
    if not display_matrix_valid:
        st.warning(
            "The Flow Map is showing the Origin-Constrained fallback because IPF failed. "
            "No all-zero matrix is presented as a valid traffic result."
        )
    elif destination_scale_factor != 1.0:
        st.warning(
            f"Destination targets were explicitly scaled by {destination_scale_factor:.8f} "
            "to match the origin total before IPF."
        )

fig_zones = plot_zones(zones)
fig_flows = plot_flows(
    display_matrix,
    mode_sel,
    constraint_sel,
    percentile,
    coords_mat,
    zones,
    highlighted_od=(int(origin_i), int(destination_j)) if teaching_enabled else None,
    matrix_label=display_matrix_label,
)
fig_decay = plot_decay(beta_sliders, fitted_results, modes)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🗺️ Zone Distribution")
    st.plotly_chart(fig_zones, use_container_width=True)
with col2:
    st.markdown("### 🚗 Flow Map")
    if teaching_enabled:
        st.caption(
            f"Highlighted OD {selected_snapshot['origin']} → {selected_snapshot['destination']} · "
            f"{selected_snapshot['flow']:,.6f} {display_value_unit} · {display_matrix_label}"
        )
    st.plotly_chart(fig_flows, use_container_width=True)

st.markdown("### 📈 Distance Decay Fit (True β from sliders)")
st.plotly_chart(fig_decay, use_container_width=True)

st.markdown(
    """
    **Explanation:**
    - 🟦 **Theoretical** (blue solid line): The *theoretical* distance decay function defined by the slider β value  
    - 🔴 **Fitted** (red dashed line): The *estimated* decay function obtained from the currently displayed simulation data  
    - ⚫ **Power-law tail ∝ d⁻¹·⁵** (gray dotted line): A *reference* curve illustrating a long-tail decay pattern

    The fitted β is a diagnostic only. Switching constraints never replaces the slider β.
    """,
    unsafe_allow_html=True,
)


# ------------------------------
# Doubly-constrained teaching playback
# ------------------------------
if constraint_sel == "Doubly-Constrained" and show_balance_process:
    with st.expander("⚖️ Doubly-Constrained Balancing Process", expanded=True):
        if ipf_result is None:
            st.error("No IPF result is available.")
        elif not ipf_result.converged:
            st.error(ipf_result.message)
            st.metric("Iteration status", ipf_result.status)
            st.metric("Maximum error", f"{ipf_result.error:.8g}")
            st.warning(
                "Playback is unavailable because there is no converged double-constrained "
                "result. The Flow Map remains on the Origin-Constrained fallback."
            )
        else:
            last_frame = len(ipf_result.history) - 1
            frame_index = min(int(st.session_state.ipf_frame), last_frame)

            def set_ipf_frame(new_frame):
                new_frame = int(np.clip(new_frame, 0, last_frame))
                st.session_state.ipf_frame = new_frame
                st.session_state.ipf_autoplay = False

            def toggle_ipf_autoplay():
                st.session_state.ipf_autoplay = not st.session_state.ipf_autoplay

            def stop_ipf_autoplay():
                st.session_state.ipf_autoplay = False

            playback_mode = st.radio(
                "Playback mode",
                ["Full frame-by-frame", "Fast classroom demo"],
                horizontal=True,
                key="ipf_playback_mode",
                on_change=stop_ipf_autoplay,
                help=(
                    "Fast mode uses only real saved frames: the first row adjustment, "
                    "completed iterations 1, 2, 5, 10, 20, 50, and the final frame."
                ),
            )
            fast_frames = fast_playback_frames(ipf_result.history)
            playback_frames = (
                list(range(len(ipf_result.history)))
                if playback_mode == "Full frame-by-frame"
                else fast_frames
            )
            previous_frame = playback_neighbor(frame_index, playback_frames, -1)
            next_frame = playback_neighbor(frame_index, playback_frames, 1)

            controls = st.columns([1, 1, 1, 1, 1.3, 2.5])
            controls[0].button(
                "Reset", on_click=set_ipf_frame, args=(0,), use_container_width=True
            )
            controls[1].button(
                "← Previous", on_click=set_ipf_frame,
                args=(previous_frame,), use_container_width=True,
            )
            controls[2].button(
                "Next →", on_click=set_ipf_frame,
                args=(next_frame,), use_container_width=True,
            )
            controls[3].button(
                "Final", on_click=set_ipf_frame, args=(last_frame,), use_container_width=True
            )
            controls[4].button(
                "Pause" if st.session_state.ipf_autoplay else "Auto play",
                on_click=toggle_ipf_autoplay,
                use_container_width=True,
            )
            controls[5].progress(
                frame_index / max(last_frame, 1), text=f"Frame {frame_index} / {last_frame}"
            )
            jump_key = f"ipf_jump_request_{playback_signature}"

            def jump_to_requested_frame():
                set_ipf_frame(st.session_state[jump_key])

            jump_cols = st.columns([4, 1])
            jump_cols[0].slider(
                "Quick jump to any real history frame",
                min_value=0,
                max_value=last_frame,
                value=0,
                key=jump_key,
                help=(
                    "This slider is a jump request, separate from the current playback frame. "
                    "Press Jump to apply it."
                ),
            )
            jump_cols[1].button(
                "Jump", on_click=jump_to_requested_frame, use_container_width=True
            )
            if playback_mode == "Fast classroom demo":
                fast_labels = []
                for index in fast_frames:
                    item = ipf_result.history[index]
                    fast_labels.append(
                        "initial" if item['phase'] == 'initial'
                        else f"iteration {item['iteration']} {item['phase']}"
                    )
                st.caption("Fast sequence (real history only): " + " → ".join(fast_labels))

            frame_state = playback_frame_snapshot(ipf_result.history, frame_index)
            frame = ipf_result.history[frame_index]
            phase_name = {
                "initial": "Initial weights / 初始权重",
                "row": "Row adjustment",
                "column": "Column adjustment",
            }[frame['phase']]
            st.markdown(
                f"#### {phase_name} · true iteration {frame_state['iteration']}"
            )
            if frame_state['unit'] == 'unitless weight':
                st.warning("初始权重（无出行次数单位） / Initial weights have no trip-count unit.")
            else:
                st.success("OD 矩阵单位：次出行 / OD matrix unit: trips.")
            st.caption(
                "Row adjustment applies Oᵢ / ΣⱼFᵢⱼ. Column adjustment applies "
                "Dⱼ / ΣᵢFᵢⱼ. Values are stored as floats; display rounding is not fed back."
            )
            i, j = int(origin_i), int(destination_j)
            current = selected_od_snapshot(frame['matrix'], i, j)
            initial = selected_od_snapshot(ipf_result.history[0]['matrix'], i, j)
            final = selected_od_snapshot(ipf_result.matrix, i, j)
            adjustment_factor = 1.0
            factor_label = "No adjustment"
            if frame_index > 0:
                previous_matrix = ipf_result.history[frame_index - 1]['matrix']
                if frame['phase'] == 'row':
                    previous_total = previous_matrix[i, :].sum()
                    adjustment_factor = origin_targets[i] / previous_total if previous_total > 0 else np.nan
                    factor_label = f"Row factor for origin {i}"
                elif frame['phase'] == 'column':
                    previous_total = previous_matrix[:, j].sum()
                    adjustment_factor = destination_targets[j] / previous_total if previous_total > 0 else np.nan
                    factor_label = f"Column factor for destination {j}"

            flow_cols = st.columns(4)
            flow_cols[0].metric(
                "Initial selected weight (unitless)", f"{initial['flow']:,.6f}"
            )
            flow_cols[1].metric(
                f"Current selected OD ({frame_state['unit']})",
                f"{current['flow']:,.6f}",
            )
            flow_cols[2].metric("Final selected OD (trips)", f"{final['flow']:,.6f}")
            flow_cols[3].metric(factor_label, f"{adjustment_factor:.8f}")

            total_cols = st.columns(4)
            total_cols[0].metric(f"Current row Σ (origin {i})", f"{current['row_sum']:,.6f}")
            total_cols[1].metric("Origin target Oᵢ", f"{origin_targets[i]:,.6f}")
            total_cols[2].metric(
                f"Current column Σ (destination {j})", f"{current['column_sum']:,.6f}"
            )
            total_cols[3].metric("Destination target Dⱼ", f"{destination_targets[j]:,.6f}")

            with st.expander(f"Full {N}×{N} matrix details"):
                st.dataframe(
                    matrix_with_totals(frame['matrix']).style.format("{:.3f}"),
                    use_container_width=True,
                    height=430,
                )
            metric_cols = st.columns(4)
            metric_cols[0].metric("True iteration", frame_state['iteration'])
            metric_cols[1].metric("Row max error", f"{frame_state['row_error']:.8g}")
            metric_cols[2].metric("Column max error", f"{frame_state['col_error']:.8g}")
            metric_cols[3].metric(
                "Current status", "Converged" if frame_index == last_frame else "In progress"
            )
            st.caption(f"Current maximum error: {frame_state['error']:.8g}")

            convergence = pd.DataFrame({
                'frame': np.arange(len(ipf_result.history)),
                'maximum error': [item['error'] for item in ipf_result.history],
                'phase': [item['phase'] for item in ipf_result.history],
                'iteration': [item['iteration'] for item in ipf_result.history],
            })
            with st.expander("Convergence history"):
                st.line_chart(convergence.set_index('frame')['maximum error'])
                st.dataframe(convergence, use_container_width=True, height=220)

            if st.session_state.ipf_autoplay:
                next_auto_frame = playback_neighbor(
                    st.session_state.ipf_frame, playback_frames, 1
                )
                if next_auto_frame != st.session_state.ipf_frame:
                    time.sleep(0.65)
                    st.session_state.ipf_frame = next_auto_frame
                    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun")
                    rerun()
                else:
                    st.session_state.ipf_autoplay = False
