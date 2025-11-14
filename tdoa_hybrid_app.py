# tdoa_hybrid_app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import least_squares
import time
import pandas as pd

st.set_page_config(page_title="TDOA Hybrid Position Error", layout="wide")
st.title("📡 TDOA Position Error Tool — Hybrid Model (Isochrones + Heatmap)")

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
C = 3e8  # Speed of light (m/s)


# -------------------------------------------------
# MODEL FUNCTIONS
# -------------------------------------------------
def tdoa_vector(x, sensors, ref_idx=0):
    """
    Compute range-difference vector for all sensors relative to reference.
    Returns vector of length N.
    """
    x = np.asarray(x)
    s = np.asarray(sensors)

    # scalar dist from x to reference
    d_ref = np.linalg.norm(x - s[ref_idx])

    # vector distances from x to all sensors
    d_all = np.linalg.norm(x - s, axis=1)

    return d_all - d_ref


def jacobian_tdoa(x, sensors, ref_idx=0):
    """
    Jacobian of TDOA measurement model.
    J[i] = ∂/∂x ( |x - s_i| - |x - s_ref| )
    Returns (N-1, 2)
    """
    x = np.asarray(x)
    s = np.asarray(sensors)
    N = len(s)

    r = np.linalg.norm(x - s, axis=1)
    r[r == 0] = 1e-12

    grad = np.zeros((N - 1, 2))
    for i in range(N):
        if i == ref_idx:
            continue
        idx = i if i < ref_idx else i - 1
        grad[idx, :] = (x - s[i]) / r[i] - (x - s[ref_idx]) / r[ref_idx]

    return grad


@st.cache_data(show_spinner=False)
def compute_dop_grid(sensors, ref_idx, xs, ys, time_error_s):
    """
    Fast DOP-based error heatmap.
    Returns (XX, YY, error_grid).
    """
    sensors = np.asarray(sensors)
    XX, YY = np.meshgrid(xs, ys)
    out = np.zeros_like(XX)

    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            p = np.array([XX[i, j], YY[i, j]])
            J = jacobian_tdoa(p, sensors, ref_idx)

            # covariance ~ (Cσ)^2 (JᵀJ)^-1
            JTJ = J.T @ J + np.eye(2) * 1e-12
            try:
                cov = (C * time_error_s) ** 2 * np.linalg.inv(JTJ)
                out[i, j] = np.sqrt(np.trace(cov))
            except np.linalg.LinAlgError:
                out[i, j] = np.nan

    return XX, YY, out


def multilateration_least_squares(initial, sensors, meas):
    """
    Solve nonlinear TDOA multilateration with least squares.
    meas is length N-1 (sensor 1 vs others).
    """
    s = np.asarray(sensors)
    ref_idx = 0

    def fun(x):
        td = tdoa_vector(x, s, ref_idx)[1:]
        return td - meas

    result = least_squares(fun, x0=initial, method="lm")
    return result.x


def simulate_monte_carlo(target, sensors, ref_idx, time_error_s, trials=400):
    """
    Estimate RMSE via Monte-Carlo at a single target location.
    """
    sensors = np.asarray(sensors)
    true_rd = tdoa_vector(target, sensors, ref_idx)
    meas_true = true_rd[1:]

    est_positions = []

    for _ in range(trials):
        # timing noise per sensor
        noise_s = np.random.normal(scale=time_error_s, size=len(sensors))
        rd_noise = (noise_s - noise_s[ref_idx]) * C

        noisy = true_rd + rd_noise
        noisy_meas = noisy[1:]

        try:
            est = multilateration_least_squares(target, sensors, noisy_meas)
            est_positions.append(est)
        except:
            pass

    if len(est_positions) == 0:
        return np.nan

    est_positions = np.array(est_positions)
    errs = np.linalg.norm(est_positions - target, axis=1)

    return np.sqrt(np.mean(errs ** 2))


# -------------------------------------------------
# SIDEBAR INPUTS
# -------------------------------------------------
st.sidebar.header("Sensor Locations (meters)")
num_sensors = st.sidebar.number_input("Number of sensors", 4, 12, 4)

default_positions = [
    (-5000, 0),
    (5000, 0),
    (0, 8000),
    (8000, 8000)
]

sensors = []
for i in range(num_sensors):
    default = default_positions[i] if i < len(default_positions) else (1000 * i, 0)
    c = st.sidebar.columns(2)
    x = c[0].number_input(f"S{i+1} X", value=float(default[0]), key=f"sx{i}")
    y = c[1].number_input(f"S{i+1} Y", value=float(default[1]), key=f"sy{i}")
    sensors.append((x, y))

ref_idx = st.sidebar.selectbox("Reference sensor", list(range(1, num_sensors + 1)), index=0) - 1

st.sidebar.header("Target")
tc = st.sidebar.columns(2)
tx = tc[0].number_input("Target X", value=0.0)
ty = tc[1].number_input("Target Y", value=10000.0)
target = np.array([tx, ty])

st.sidebar.header("Timing Error")
time_error_ns = st.sidebar.slider("Timing error (ns)", 0, 500, 50)
time_error_s = time_error_ns * 1e-9

st.sidebar.header("Grid")
grid_km = st.sidebar.slider("Grid half-size (km)", 5, 30, 15)
grid_res = st.sidebar.slider("Grid resolution", 80, 300, 160)

xs = np.linspace(target[0] - grid_km * 1000, target[0] + grid_km * 1000, grid_res)
ys = np.linspace(target[1] - grid_km * 1000, target[1] + grid_km * 1000, grid_res)


# -------------------------------------------------
# COMPUTE HEATMAP
# -------------------------------------------------
with st.spinner("Computing DOP heatmap..."):
    XX, YY, dop = compute_dop_grid(sensors, ref_idx, xs, ys, time_error_s)


# -------------------------------------------------
# COMPUTE ISOCHRONES
# -------------------------------------------------
true_rd = tdoa_vector(target, sensors, ref_idx)
dR = C * time_error_s

# -------------------------------------------------
# PLOTLY FIGURE
# -------------------------------------------------
fig = go.Figure()

# ---- Heatmap ----
fig.add_trace(go.Heatmap(
    x=xs / 1000.0,
    y=ys / 1000.0,
    z=dop,
    colorscale="YlOrRd",
    colorbar=dict(title="Error (m)"),
    name="DOP Heatmap"
))

# ---- Isochrones ----
sensors_arr = np.asarray(sensors)

for i in range(num_sensors):
    if i == ref_idx:
        continue

    Ri = np.sqrt((XX - sensors_arr[i, 0]) ** 2 + (YY - sensors_arr[i, 1]) ** 2)
    Rref = np.sqrt((XX - sensors_arr[ref_idx, 0]) ** 2 + (YY - sensors_arr[ref_idx, 1]) ** 2)
    Z = Ri - Rref

    levels = [true_rd[i], true_rd[i] + dR, true_rd[i] - dR]

    # central contour (solid)
    fig.add_trace(go.Contour(
        x=xs / 1000.0,
        y=ys / 1000.0,
        z=Z,
        contours=dict(values=[levels[0]]),
        line=dict(width=3, color="blue"),
        showscale=False,
        hoverinfo="skip",
        name=f"Isochrone S{i+1}"
    ))

    # +dR
    fig.add_trace(go.Contour(
        x=xs / 1000.0,
        y=ys / 1000.0,
        z=Z,
        contours=dict(values=[levels[1]]),
        line=dict(width=1, color="royalblue", dash="dot"),
        showscale=False,
        hoverinfo="skip"
    ))

    # -dR
    fig.add_trace(go.Contour(
        x=xs / 1000.0,
        y=ys / 1000.0,
        z=Z,
        contours=dict(values=[levels[2]]),
        line=dict(width=1, color="royalblue", dash="dot"),
        showscale=False,
        hoverinfo="skip"
    ))

# ---- Sensors and Target ----
fig.add_trace(go.Scatter(
    x=sensors_arr[:, 0] / 1000.0,
    y=sensors_arr[:, 1] / 1000.0,
    mode="markers+text",
    text=[f"S{i+1}" for i in range(num_sensors)],
    textposition="top center",
    marker=dict(size=10, color="black", symbol="square"),
    name="Sensors"
))

fig.add_trace(go.Scatter(
    x=[target[0] / 1000.0],
    y=[target[1] / 1000.0],
    mode="markers+text",
    text=["Target"],
    textposition="bottom center",
    marker=dict(size=14, color="green"),
    name="Target"
))

fig.update_layout(
    title="TDOA Isochrones + Error Heatmap",
    xaxis_title="X (km)",
    yaxis_title="Y (km)",
    height=850,
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)

st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------
# Monte-Carlo RMSE at target
# -------------------------------------------------
st.subheader("Monte-Carlo RMSE at Target")
with st.spinner("Running Monte-Carlo…"):
    rmse = simulate_monte_carlo(target, sensors, ref_idx, time_error_s, trials=300)

st.success(f"RMSE at target ≈ **{rmse:.1f} m** (300 trials)")
