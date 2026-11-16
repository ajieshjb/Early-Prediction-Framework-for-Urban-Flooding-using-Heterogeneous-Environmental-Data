"""Flood early-warning demo.  Run:  streamlit run app.py"""
import json, os
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import urban_flood as uf
from flood_service import FloodSystem

st.set_page_config(page_title="Urban Flood Early Warning", page_icon="🌧️", layout="wide")


@st.cache_resource(show_spinner="Building the synthetic city and training the model (about 30-60 s, once)...")
def get_system():
    return FloodSystem(seed=42)


def draw_map(values, title, cmap, vmin, vmax, flagged=None, labels=None):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    grid = values.reshape(uf.GH, uf.GW)
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
    for y in range(uf.GH):
        for x in range(uf.GW):
            c = y * uf.GW + x
            txt = labels[c] if labels is not None else f"{grid[y, x]:.2f}"
            ax.text(x, y, txt, ha="center", va="center", fontsize=8,
                    color="white" if grid[y, x] > (vmin + vmax) / 2 else "black")
            if flagged is not None and flagged[c]:
                ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, fill=False, ec="red", lw=2.5))
    ax.set_xticks(range(uf.GW)); ax.set_yticks(range(uf.GH))
    ax.set_xlabel("river is along the top row (y = 0)", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return fig


sysm = get_system()
st.title("🌧️ Urban Flood Early Warning")
st.caption("Demo on SYNTHETIC data (24-cell city). Predictor shown: Random Forest. "
           "Red outline = cell predicted to flood (depth above 0.15 m).")

tab_live, tab_perf = st.tabs(["Live warning demo", "Model performance"])

with tab_live:
    with st.sidebar:
        st.header("Controls")
        lead = st.radio("Lead time (hours ahead)", uf.HORIZONS, index=1, horizontal=True)
        h_idx = uf.HORIZONS.index(lead)
        default_hour = sysm.warning_hour(h_idx)
        if "hour" not in st.session_state:
            st.session_state.hour = default_hour
        if st.button("Jump to an early-warning event"):
            st.session_state.hour = default_hour
        st.caption("The jump button picks a well-predicted event for the demo. "
                   "Drag the slider to see misses and false alarms too.")
        hour = st.slider("Time in the test period (hour)", 0, sysm.n_test_hours - 1, key="hour")
        st.subheader("What-if scenario")
        rain_mult = st.slider("Rainfall multiplier", 0.0, 3.0, 1.0, 0.1)
        river_add = st.slider("Extra river level", 0.0, 4.0, 0.0, 0.1)

    seq = sysm.scenario(sysm.window(hour), rain_mult, river_add)
    prob, depth, flag, thr = sysm.predict(seq, h_idx)
    n_risk = int(flag.sum())
    recent_rain = float(seq[:, -3:, 0].mean(0).mean())

    if n_risk == 0:
        st.success(f"No alert: no cell is predicted to flood in the next {lead} h.")
    elif n_risk < 6:
        st.warning(f"WATCH: {n_risk} of {uf.N_CELLS} cells may flood within {lead} h.")
    else:
        st.error(f"WARNING: {n_risk} of {uf.N_CELLS} cells likely to flood within {lead} h.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Cells at risk", f"{n_risk} / {uf.N_CELLS}")
    c2.metric("Max predicted depth", f"{depth.max():.2f} m")
    c3.metric("Rain, last 3 h (mean)", f"{recent_rain:.1f} mm/h")

    left, right = st.columns(2)
    labels = [f"{c}\n{prob[c]*100:.0f}%" for c in range(uf.N_CELLS)]
    with left:
        st.pyplot(draw_map(prob, f"Flood risk in {lead} h (probability, cell id)", "YlOrRd", 0, 1, flag, labels))
    with right:
        if rain_mult == 1.0 and river_add == 0.0:
            a_flood, a_depth = sysm.actual(hour, h_idx)
            st.pyplot(draw_map(a_depth, f"What actually happened at +{lead} h (depth, m)", "Blues",
                               0, max(0.5, float(a_depth.max())), a_flood > 0))
            hits = int(((a_flood > 0) & flag).sum())
            st.caption(f"Truly flooded: {int(a_flood.sum())} cells. Correctly flagged: {hits}. "
                       f"Missed: {int(((a_flood > 0) & ~flag).sum())}. False alarms: {int((~(a_flood > 0) & flag).sum())}.")
        else:
            st.pyplot(draw_map(depth, f"Predicted depth at +{lead} h (m), what-if scenario", "Blues",
                               0, max(0.5, float(depth.max()))))
            st.caption("Scenario mode: there is no ground truth for an imagined storm.")

    order = np.argsort(-prob)[:5]
    st.subheader("Top 5 riskiest cells")
    st.dataframe({"cell": order.tolist(), "row (y)": (order // uf.GW).tolist(), "col (x)": (order % uf.GW).tolist(),
                  "flood probability": np.round(prob[order], 2).tolist(),
                  "predicted depth (m)": np.round(depth[order], 2).tolist()}, hide_index=True)
    st.caption(f"Alert threshold for {lead} h = {thr:.2f} (tuned on the validation period).")

with tab_perf:
    st.subheader("Test-set results (synthetic data)")
    path = os.path.join("outputs", "results.json")
    if os.path.exists(path):
        res = json.load(open(path))["results"]
        rows = [{"Model": r["model"], **{f"F1 {h}h": round(v, 3) for h, v in zip(uf.HORIZONS, r["f1"])},
                 **{f"RMSE {h}h (m)": round(v, 3) for h, v in zip(uf.HORIZONS, r["rmse"])}} for r in res]
        st.dataframe(rows, hide_index=True)
    else:
        st.info("Run `python urban_flood.py` first to create outputs/results.json.")
    for img, cap in [("lead_time_f1.png", "F1 by lead time"), ("training_curve.png", "Training curves")]:
        p = os.path.join("outputs", img)
        if os.path.exists(p):
            st.image(p, caption=cap, width=520)
