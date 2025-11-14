# tdoa_hybrid_app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import least_squares
from functools import lru_cache
import time

st.set_page_config(page_title="TDOA Hybrid: Isochrones + Heatmap", layout="wide")
st.title("🔭 TDOA (4+ sensor) — Hybrid Isochrones & Error Heatmap")

# -------------------------
# Utility / Model functions
# -------------------------
C = 3e8  # speed of light (m/s)

def tdoa_vector(x, sensors, ref_idx=0):
    """
    Given a candidate position x (2,), compute TDOA vector (range differences)
    relative to reference sensor (ref_idx). Returns vector of length (N-1,).
    """
    x = np.asarray(x)
    s = np.asarray(sensors)
    d_ref = np.linalg.norm(x - s[ref_idx], axis=1)  # distances to ref duplicated
    # distances to each sensor
    d_all = np.linalg.norm(x - s, axis=1)
    return d_all - d_all[ref_idx]

def jacobian_tdoa(x, sensors, ref_idx=0):
    """
    Jacobian of h_i(x) = |x - s_i| - |x - s_ref|.
    Returns matrix shape ((N-1), 2)
    """
    x = np.asarray(x)
    s = np.asarray(sensors)
    N = len(s)
    r = np.linalg.norm(x - s, axis=1)
    # Avoid division by zero
    r[r == 0] = 1e-12
    grad = np.zeros((N-1, 2))
    ref = ref_idx
    for i in range(N):
        if i == ref:
            continue
        idx = i if i < ref else i - 1
        grad[idx, :] = (x - s[i]) / r[i] - (x - s[ref]) / r[ref]
    return grad

@st.cache_data(show_spinner=False)
def compute_dop_grid(sensors, ref_idx, grid_x, grid_y, time_error_s):
    """
    Compute a fast DOP-based position error estimate for each grid point.
    We'll use approximate covariance: cov = (c * dt)^2 * (J^T J)^{-1}
    and derive scalar error as sqrt(trace(cov)) (RMS radius).
    """
    sx = np.array(sensors)
    nx = len(grid_x)
    ny = len(grid_y)
    XX, YY = np.meshgrid(grid_x, grid_y)
    points = np.vstack([XX.ravel(), YY.ravel()]).T
    Npts = points.shape[0]

    errors = np.zeros(Npts, dtype=float)
    small = 1e-12
    for idx, p in enumerate(points):
        J = jacobian_tdoa(p, sx, ref_idx)  # (N-1,2)
        # compute (J^T J)
        JTJ = J.T @ J
        # regularize
        JTJ += np.eye(2) * small
        try:
            cov = (C * time_error_s)**2 * np.linalg.inv(JTJ)
            # scalar error measure
            errors[idx] = np.sqrt(np.trace(cov))
        except np.linalg.LinAlgError:
            errors[idx] = np.nan
    return XX, YY, errors.reshape(YY.shape)

def multilateration_least_squares(initial, sensors, meas):
    """
    Solve for position that minimizes || (|x - s_i| - |x - s_ref|) - meas_i ||.
    meas vector is length N-1 (range-diff measurements).
    """
    s = np.asarray(sensors)
    ref_idx = 0
    def fun(x):
        return tdoa_vector(x, s, ref_idx)[1:] - meas  # exclude ref term (zero)
    res = least_squares(fun, x0=initial, method='lm')
    return res.x

def simulate_monte_carlo(target, sensors, ref_idx, time_error_s, trials=500):
    """
    Monte Carlo RMSE at the target location.
    Simulate noisy TDOA measurements and solve multilateration each trial.
    Returns RMSE (meters) across trials.
    """
    s = np.asarray(sensors)
    # true noiseless range-diffs (including ref element)
    true_rd = tdoa_vector(target, s, ref_idx)  # length N
    # use N-1 meas (exclude ref)
    true_meas = true_rd[1:]
    est_positions = []
    for _ in range(trials):
        # add Gaussian timing noise per sensor (std = time_error_s)
        # TDOA measurement noise ~ difference of two sensor timing errors
        # For simplicity, we model each measurement perturbed by normal(0, sqrt(2)*time_error_s)
        meas_noise_s = np.random.normal(scale=time_error_s, size=s.shape[0])
        rd_noise = (meas_noise_s - meas_noise_s[ref_idx]) * C  # range-diff noise (m)
        noisy_meas = true_rd + rd_noise
        noisy_meas_vec = noisy_meas[1:]
        # Solve for position (initial guess = true target)
        try:
            est = multilateration_least_squares(target, s, noisy_meas_vec)
            est_positions.append(est)
        except Exception:
            continue
    est_positions = np.array(est_positions)
    if est_positions.size == 0:
        return np.nan
    errs = np.linalg.norm(est_positions - target, axis=1)
    return np.sqrt(np.mean(errs**2))

# -------------------------
# Sidebar: inputs
# -------------------------
st.sidebar.header("Sensors (meters) — edit and press Enter")
st.sidebar.markdown("Minimum 4 sensors; edit X,Y and press Enter to apply.")

# Provide 4 sensors by default (can add more)
default_sensors = [
    (-5000.0, 0.0),
    (5000.0, 0.0),
    (0.0, 8000.0),
    (8000.0, 8000.0)
]

# allow user to specify number of sensors
num_sensors = st.sidebar.number_input("Number of sensors", min_value=4, max_value=12, value=4, step=1)

sensors = []
for i in range(num_sensors):
    default = default_sensors[i] if i < len(default_sensors) else (1000.0*i, 0.0)
    row = st.sidebar.columns([0.5,0.5])
    x = row[0].number_input(f"S{i+1} X (m)", value=float(default[0]), key=f"sx{i}")
    y = row[1].number_input(f"S{i+1} Y (m)", value=float(default[1]), key=f"sy{i}")
    sensors.append((x, y))

ref_sensor_idx = int(st.sidebar.selectbox("Reference sensor (isochrones relative to)", options=list(range(1, num_sensors+1)), index=0)) - 1

st.sidebar.markdown("---")
st.sidebar.subheader("Target")
tcol = st.sidebar.columns(2)
tx = tcol[0].number_input("Target X (m)", value=0.0, key="tx")
ty = tcol[1].number_input("Target Y (m)", value=10000.0, key="ty")
target = np.array([tx, ty])

st.sidebar.markdown("---")
time_err_ns = st.sidebar.slider("Sensor timing error (± std) [ns]", 0, 500, 50)
time_err_s = time_err_ns * 1e-9

st.sidebar.markdown("---")
st.sidebar.subheader("Grid / Heatmap")
grid_km = st.sidebar.slider("Grid half-span (km)", 5, 30, 15)  # +/- this in km
grid_pts = st.sidebar.slider("Grid resolution (per axis)", 80, 300, 160, step=8)
x_min = target[0] - grid_km*1000
x_max = target[0] + grid_km*1000
y_min = target[1] - grid_km*1000
y_max = target[1] + grid_km*1000
use_monte = st.sidebar.checkbox("Use Monte-Carlo for heatmap (slow)", value=False)
mc_trials = st.sidebar.number_input("MC trials per cell (if Monte-Carlo)", min_value=50, max_value=2000, value=200, step=50)

st.sidebar.markdown("---")
st.sidebar.markdown("Plot options")
show_isochrones = st.sidebar.checkbox("Show isochrones (contours)", value=True)
show_heatmap = st.sidebar.checkbox("Show error heatmap", value=True)
show_rms_at_target = st.sidebar.checkbox("Show Monte-Carlo RMSE at target (single-point)", value=True)

# -------------------------
# Quick geometry diagnostics
# -------------------------
sensors_arr = np.array(sensors)
ref = ref_sensor_idx

colA, colB = st.columns([1, 1])
with colA:
    st.subheader("Sensors & Target")
    st.write("Reference sensor index:", ref+1)
    st.table({
        "sensor": [f"S{i+1}" for i in range(len(sensors))],
        "x (m)": sensors_arr[:,0],
        "y (m)": sensors_arr[:,1],
    })
    st.write("Target (m):", target)

with colB:
    st.subheader("Quick metrics")
    baseline = np.linalg.norm(sensors_arr - sensors_arr[0], axis=1).max()
    dists = np.linalg.norm(target - sensors_arr, axis=1)
    st.write(f"Max baseline (m): {baseline:.1f}")
    st.write("Ranges to sensors (m):", np.round(dists,1).tolist())
    st.write(f"Timing error: {time_err_ns} ns  → range-diff std ≈ {C * time_err_s:.2f} m")

# -------------------------
# Compute DOP heatmap (fast)
# -------------------------
xs = np.linspace(x_min, x_max, grid_pts)
ys = np.linspace(y_min, y_max, grid_pts)

with st.spinner("Computing DOP-based heatmap..."):
    XX, YY, dop_errors = compute_dop_grid(tuple(map(tuple, sensors)), ref, xs, ys, time_err_s)

# Optional Monte-Carlo heatmap (very slow) - implement coarse sampling only if selected
mc_heatmap = None
if use_monte:
    st.warning("Monte-Carlo heatmap selected — this will be slow. Using coarse subsampling to limit compute.")
    start = time.time()
    # We'll do Monte-Carlo for a coarse grid to keep time reasonable
    coarse_pts = min(80, grid_pts)
    xs_coarse = np.linspace(x_min, x_max, coarse_pts)
    ys_coarse = np.linspace(y_min, y_max, coarse_pts)
    MC = np.zeros((coarse_pts, coarse_pts))
    for i, xx in enumerate(xs_coarse):
        for j, yy in enumerate(ys_coarse):
            MC[j,i] = simulate_monte_carlo(np.array([xx,yy]), sensors, ref, time_err_s, trials=mc_trials)
    mc_heatmap = (xs_coarse, ys_coarse, MC)
    st.success(f"Monte-Carlo coarse heatmap done in {time.time()-start:.1f}s")

# -------------------------
# Compute isochrone contours: compute TDOA (range-diff) for true target
# and plot contours for levels = [tdoa_val, tdoa_val ± range_diff_uncertainty]
# -------------------------
# true range-diffs (m)
true_rd = tdoa_vector(target, sensors_arr, ref)  # includes ref (0)
# measurement uncertainty (range-diff) ~ C * time_err_s
dR = C * time_err_s

# create figure with Plotly
fig = go.Figure()

# heatmap layer (DOP-based)
if show_heatmap:
    # Plotly heatmap expects z shape (ny, nx)
    fig.add_trace(go.Heatmap(
        x=xs / 1000.0,  # km
        y=ys / 1000.0,
        z=dop_errors,
        colorscale="YlOrRd",
        colorbar=dict(title="Error (m)"),
        name="DOP Error (m)",
        hovertemplate="x: %{x} km<br>y: %{y} km<br>Error: %{z:.1f} m"
    ))

# Add Monte-Carlo coarse heatmap overlay if present (as separate trace)
if mc_heatmap is not None:
    xs_coarse_km = mc_heatmap[0] / 1000.0
    ys_coarse_km = mc_heatmap[1] / 1000.0
    fig.add_trace(go.Heatmap(
        x=xs_coarse_km,
        y=ys_coarse_km,
        z=mc_heatmap[2],
        colorscale="Viridis",
        opacity=0.7,
        showscale=False,
        name="MC RMSE (m)",
        hovertemplate="x: %{x} km<br>y: %{y} km<br>RMSE: %{z:.1f} m"
    ))

# Isochrones (contours) for each sensor i > ref
if show_isochrones:
    # We'll compute the contour of (r_i - r_ref) over the grid and plot contour lines
    # For each sensor i>ref, compute Z = (r_i - r_ref) and add contour lines for:
    # central (true_rd[i]) and bounds (true_rd[i] +/- dR)
    s = sensors_arr
    for i in range(len(s)):
        if i == ref:
            continue
        # compute range-diff grid
        # r_i - r_ref for all grid points
        # distances to sensor i and ref
        Ri = np.sqrt((XX - s[i,0])**2 + (YY - s[i,1])**2)
        Rref = np.sqrt((XX - s[ref,0])**2 + (YY - s[ref,1])**2)
        Z = Ri - Rref  # meters

        levels = [true_rd[i], true_rd[i] + dR, true_rd[i] - dR]
        # central contour (solid)
        fig.add_trace(go.Contour(
            x=xs/1000.0, y=ys/1000.0, z=Z,
            contours=dict(
                start=levels[0], end=levels[0], size=1e9  # hack: single level
            ),
            line=dict(width=3),
            showscale=False,
            name=f"Isochrone S{i+1}-S{ref+1}",
            hoverinfo='none',
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'blue']],
            opacity=1.0,
            showslegend=False
        ))
        # uncertainty bands (dashed, lighter)
        fig.add_trace(go.Contour(
            x=xs/1000.0, y=ys/1000.0, z=Z,
            contours=dict(
                values=[levels[1]]
            ),
            line=dict(width=1, dash='dot'),
            showscale=False,
            name=f"Isochrone +dR S{i+1}",
            hoverinfo='none',
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'royalblue']],
            opacity=0.6,
            showslegend=False
        ))
        fig.add_trace(go.Contour(
            x=xs/1000.0, y=ys/1000.0, z=Z,
            contours=dict(
                values=[levels[2]]
            ),
            line=dict(width=1, dash='dot'),
            showscale=False,
            name=f"Isochrone -dR S{i+1}",
            hoverinfo='none',
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'royalblue']],
            opacity=0.6,
            showslegend=False
        ))

# Sensors and target markers
fig.add_trace(go.Scatter(
    x=sensors_arr[:,0]/1000.0, y=sensors_arr[:,1]/1000.0,
    mode='markers+text', marker=dict(size=10, symbol='square'),
    text=[f"S{i+1}" for i in range(len(sensors_arr))],
    textposition="top center",
    name="Sensors"
))
fig.add_trace(go.Scatter(
    x=[target[0]/1000.0], y=[target[1]/1000.0],
    mode='markers+text', marker=dict(size=12, symbol='circle', color='green'),
    text=["Target"], textposition="bottom center",
    name="Target"
))

fig.update_layout(
    title="Isochrones & Error Heatmap (km)",
    xaxis_title="X (km)",
    yaxis_title="Y (km)",
    height=800,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)

# -------------------------
# Show plots
# -------------------------
st.plotly_chart(fig, use_container_width=True)

# Show colorbar legend explanation
st.markdown("**Notes:** DOP-based heatmap is a fast linearized estimate. Isochrones show central TDOA contours (solid) and ±range-diff uncertainty (dashed).")

# -------------------------
# Monte-Carlo single-point RMSE at target
# -------------------------
if show_rms_at_target:
    with st.spinner("Running Monte-Carlo at target position..."):
        mc_rmse = simulate_monte_carlo(target, sensors, ref, time_err_s, trials=mc_trials)
    if np.isnan(mc_rmse):
        st.warning("Monte-Carlo failed to converge; try increasing trials or adjusting sensor geometry.")
    else:
        st.success(f"Monte-Carlo RMSE at target: {mc_rmse:.1f} m (trials={mc_trials})")

# -------------------------
# Allow user to export a snapshot CSV of the DOP heatmap (downsampled)
# -------------------------
import pandas as pd
if st.button("Export DOP heatmap (sampled CSV)"):
    # sample to smaller CSV to keep size modest
    sample_step = max(1, grid_pts // 120)
    xs_s = xs[::sample_step] / 1000.0
    ys_s = ys[::sample_step] / 1000.0
    XXs, YYs = np.meshgrid(xs_s, ys_s)
    Zs = dop_errors[::sample_step, ::sample_step]
    df = pd.DataFrame({
        "x_km": XXs.ravel(),
        "y_km": YYs.ravel(),
        "error_m": Zs.ravel()
    })
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, file_name="tdoa_dop_heatmap_sampled.csv", mime="text/csv")
