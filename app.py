"""
National Foods (Company Y) — Demand-Driven Production Scheduling System (v2)
Dedicated-line architecture: each product runs on its own line, changing over
between its variants. Date-range planning with per-variant demand forecasting.

Loads artifacts exported by the Kaggle notebook:
  lstm_demand.keras, xscaler.joblib, yscaler.joblib, config.joblib
Falls back to a seasonal-naive forecaster if the model files are absent.

Run:    streamlit run app.py
Deploy: push repo to GitHub -> Streamlit Community Cloud
"""
import os, random
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Company Y — Production Scheduler", page_icon="🌾", layout="wide")

# ============================================================================
# LINE & VARIANT STRUCTURE
# Each product = one dedicated line running its variants in parallel with the
# others. Changeovers occur WITHIN a line, between its variants only.
# Line capacities are calibrated from the expert interview (see README).
# ============================================================================
LINES = {
    "Maize Meal": {"code": "MZ-RS-10", "capacity_tpd": 480.0, "color": "#1F4E79",
        "variants": {
            "Refined Mealie-Meal":     {"share": 0.30, "changeover_hr": 3.0},
            "Roller Meal":             {"share": 0.55, "changeover_hr": 2.5},
            "Multi-Grain Mealie-Meal": {"share": 0.15, "changeover_hr": 4.0}}},
    "Flour": {"code": "FL-SW-50", "capacity_tpd": 300.0, "color": "#5B9BD5",
        "variants": {
            "White Flour": {"share": 0.82, "changeover_hr": 2.0},
            "Bran Flour":  {"share": 0.18, "changeover_hr": 3.0}}},
    "Rice": {"code": "RC-MH-25", "capacity_tpd": 150.0, "color": "#C00000",
        "variants": {
            "White Rice": {"share": 0.80, "changeover_hr": 1.5},
            "Brown Rice": {"share": 0.20, "changeover_hr": 2.0}}},
    "Pasta": {"code": "PA-BB-05", "capacity_tpd": 60.0, "color": "#2E7D32",
        "variants": {
            "White Pasta": {"share": 0.70, "changeover_hr": 2.5},
            "Brown Pasta": {"share": 0.30, "changeover_hr": 3.0}}},
}
HOLD = {"Maize Meal": 0.45, "Flour": 0.55, "Rice": 0.40, "Pasta": 0.50}
STOCKOUT = {"Maize Meal": 180, "Flour": 220, "Rice": 150, "Pasta": 140}
CO_RATE_DEFAULT = 450.0
PRODUCTS = list(LINES.keys())
CODE = {L: LINES[L]["code"] for L in LINES}

# ============================================================================
# ARTIFACT + DATA LOADING
# ============================================================================
@st.cache_resource
def load_model():
    try:
        import joblib
        from tensorflow.keras.models import load_model as lm
        if all(os.path.exists(f) for f in ["lstm_demand.keras", "xscaler.joblib", "yscaler.joblib"]):
            return lm("lstm_demand.keras", compile=False), joblib.load("xscaler.joblib"), joblib.load("yscaler.joblib")
    except Exception as e:
        st.session_state["model_warn"] = str(e)
    return None, None, None

@st.cache_data
def load_history():
    for p in ["nationalfoods_demand_2023_2025.csv", "data/nationalfoods_demand_2023_2025.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p, parse_dates=["date"]).sort_values(["product_code", "date"]).reset_index(drop=True)
    return None

model, xscaler, yscaler = load_model()
hist = load_history()

FEATURES = ["lag1", "lag7", "lag14", "roll7", "roll7_std", "roll30", "sin_doy", "cos_doy",
            "dow", "is_school_term", "is_public_holiday", "is_festive", "is_month_end",
            "is_promo", "drought_index", "unit_price_usd_per_kg"]

def make_features(df):
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek; df["doy"] = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2*np.pi*df["doy"]/365); df["cos_doy"] = np.cos(2*np.pi*df["doy"]/365)
    out = []
    for pid, g in df.groupby("product_code"):
        g = g.sort_values("date").copy(); y = g["demand_orders_tonnes"]
        for L in (1, 7, 14): g[f"lag{L}"] = y.shift(L)
        g["roll7"] = y.shift(1).rolling(7).mean(); g["roll7_std"] = y.shift(1).rolling(7).std()
        g["roll30"] = y.shift(1).rolling(30).mean(); out.append(g)
    return pd.concat(out).reset_index(drop=True)

def forecast_line_daily(line, horizon):
    """Forecast total daily demand for a line's product over the horizon (list length=horizon)."""
    pid = CODE[line]
    if model is not None and xscaler is not None and yscaler is not None and hist is not None:
        feat = make_features(hist)
        pcols = [f"p_{p}" for p in CODE.values()]
        dummies = pd.get_dummies(feat["product_code"], prefix="p")
        for c in pcols:
            if c not in dummies: dummies[c] = 0
        feat = pd.concat([feat, dummies[pcols].astype(float)], axis=1)
        allx = FEATURES + pcols
        feat = feat.dropna(subset=FEATURES)
        g = feat[feat.product_code == pid].sort_values("date")
        if len(g) >= 14:
            window = xscaler.transform(g[allx].values[-14:]).copy()
            preds = []
            for _ in range(horizon):
                ps = model.predict(window[np.newaxis, :, :], verbose=0).flatten()[0]
                preds.append(max(float(yscaler.inverse_transform([[ps]])[0, 0]), 0))
                window = np.vstack([window[1:], window[-1]])
            return preds
    # fallback: seasonal-naive (last 7-day mean)
    if hist is not None:
        m = float(hist[hist.product_code == pid]["demand_orders_tonnes"].tail(7).mean())
    else:
        m = LINES[line]["capacity_tpd"]*0.8
    return [m]*horizon

# ============================================================================
# DEDICATED-LINE SCHEDULER  (one changeover per variant; urgency ordering)
# ============================================================================
def schedule_line(line, daily_total, opening_stock, co_rate):
    """daily_total: list of forecast daily demand for the LINE over the horizon.
    Splits to variants by share, schedules contiguous blocks, returns plan+costs."""
    H = len(daily_total)
    cap = LINES[line]["capacity_tpd"]; variants = LINES[line]["variants"]
    # per-variant daily demand = line daily * share (time-varying)
    vd = {v: [daily_total[d]*variants[v]["share"] for d in range(H)] for v in variants}
    vmean = {v: float(np.mean(vd[v])) for v in variants}

    safety_days = 1.0
    need = {v: max(sum(vd[v]) + vmean[v]*safety_days - opening_stock.get(v, 0), 0) for v in variants}

    def cover(v): return opening_stock.get(v, 0)/vmean[v] if vmean[v] > 0 else 1e9
    order = sorted(variants, key=cover)

    plan = {v: [0.0]*H for v in variants}
    day = 0; sequence = []; co_hours = 0.0; feasible = True
    for v in order:
        rem = need[v]
        if rem <= 1e-6: continue
        sequence.append(v); co_hours += variants[v]["changeover_hr"]
        while rem > 1e-6 and day < H:
            prod = min(cap, rem); plan[v][day] += prod; rem -= prod; day += 1
        if rem > 1e-6: feasible = False

    co_cost = co_hours*co_rate; hold_c = so_c = 0.0
    inv = {v: [] for v in variants}; unmet_tot = 0.0
    for v in variants:
        stock = opening_stock.get(v, 0)
        for d in range(H):
            avail = stock + plan[v][d]; disp = min(vd[v][d], avail)
            unmet = max(vd[v][d]-disp, 0); stock = avail-disp
            so_c += unmet*STOCKOUT[line]; hold_c += stock*HOLD[line]
            unmet_tot += unmet; inv[v].append(stock)
    total = co_cost+hold_c+so_c
    util = sum(need.values())/(cap*H) if cap*H > 0 else 0
    return dict(plan=plan, vd=vd, sequence=sequence, co_hours=co_hours, co_cost=co_cost,
                hold_cost=hold_c, stockout_cost=so_c, total=total, util=util,
                feasible=feasible, inv=inv, unmet=unmet_tot)

def baseline_line(line, daily_total, opening_stock, co_rate):
    """Reactive baseline: variants run in arbitrary order, switching every day (many changeovers)."""
    H = len(daily_total); variants = LINES[line]["variants"]; cap = LINES[line]["capacity_tpd"]
    vd = {v: [daily_total[d]*variants[v]["share"] for d in range(H)] for v in variants}
    plan = {v: [min(vd[v][d], cap) for d in range(H)] for v in variants}
    # daily switching among all variants -> changeover every day for each variant present
    co_hours = sum(variants[v]["changeover_hr"] for v in variants)*H/len(variants)
    co_cost = co_hours*co_rate; hold_c = so_c = 0.0
    for v in variants:
        stock = opening_stock.get(v, 0)
        for d in range(H):
            avail = stock+plan[v][d]; disp = min(vd[v][d], avail)
            unmet = max(vd[v][d]-disp, 0); stock = avail-disp
            so_c += unmet*STOCKOUT[line]; hold_c += stock*HOLD[line]
    return co_cost+hold_c+so_c

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("⚙️ Planning Controls")
st.sidebar.markdown("### Planning period")
default_start = (hist["date"].max().date() + timedelta(days=1)) if hist is not None else date.today()
start = st.sidebar.date_input("Start date", value=default_start)
end = st.sidebar.date_input("End date", value=default_start + timedelta(days=6))
if end < start:
    st.sidebar.error("End date must be on or after start date."); st.stop()
horizon = (end - start).days + 1
st.sidebar.caption(f"Planning horizon: **{horizon} day(s)**")

co_rate = st.sidebar.number_input("Changeover cost ($/hr)", 50.0, 2000.0, CO_RATE_DEFAULT, 50.0)

st.sidebar.markdown("### Line capacities & opening stock")
opening = {}
for line in PRODUCTS:
    with st.sidebar.expander(f"{line}  ({LINES[line]['capacity_tpd']:.0f} t/day)"):
        LINES[line]["capacity_tpd"] = st.number_input(f"Capacity (t/day) — {line}", 10.0, 1000.0,
            float(LINES[line]["capacity_tpd"]), 10.0, key=f"cap_{line}")
        for v in LINES[line]["variants"]:
            opening[(line, v)] = st.number_input(f"Opening stock (t) — {v}", 0.0, 5000.0,
                round(LINES[line]["capacity_tpd"]*LINES[line]["variants"][v]["share"]*horizon, 0),
                10.0, key=f"op_{line}_{v}")

run = st.sidebar.button("▶ Generate production plan", type="primary", use_container_width=True)

# ============================================================================
# HEADER
# ============================================================================
st.title("🌾 Demand-Driven Production Scheduling System")
st.caption("Company Y — dedicated-line scheduling with per-variant demand forecasting. "
           "Each product runs on its own line, changing over between its variants. "
           "Data is synthetic, calibrated to expert interview and public sources.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecaster", "LSTM (global)" if model is not None else "Seasonal-naive")
c2.metric("Lines", str(len(PRODUCTS)))
c3.metric("Total variants", str(sum(len(LINES[l]["variants"]) for l in LINES)))
c4.metric("Horizon", f"{horizon} day(s)")

if hist is None:
    st.warning("Demand history not found. Place `nationalfoods_demand_2023_2025.csv` in the app folder.")

# compute
if run or "v2_done" not in st.session_state:
    with st.spinner("Forecasting per-line demand and scheduling each line…"):
        results = {}
        for line in PRODUCTS:
            daily = forecast_line_daily(line, horizon)
            opstock = {v: opening[(line, v)] for v in LINES[line]["variants"]}
            r = schedule_line(line, daily, opstock, co_rate)
            r["daily_total"] = daily
            r["baseline"] = baseline_line(line, daily, opstock, co_rate)
            results[line] = r
        st.session_state["v2_done"] = True
        st.session_state["results"] = results
        st.session_state["dates"] = [start + timedelta(days=d) for d in range(horizon)]

results = st.session_state["results"]
dates = st.session_state["dates"]
date_labels = [d.strftime("%a %d %b") for d in dates]

tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "🏭 Production Plan", "💲 Costs", "🔬 Lines & Model"])

# ---- TAB 1: FORECAST ----
with tab1:
    st.subheader(f"Demand forecast by product variant — {start:%d %b %Y} to {end:%d %b %Y}")
    for line in PRODUCTS:
        r = results[line]
        with st.container():
            st.markdown(f"**{line}**  ·  line total forecast {np.mean(r['daily_total']):.0f} t/day")
            fig = go.Figure()
            for v in LINES[line]["variants"]:
                fig.add_trace(go.Scatter(x=date_labels, y=[round(x, 1) for x in r["vd"][v]],
                                         name=v, mode="lines+markers", stackgroup="one"))
            fig.update_layout(height=240, margin=dict(t=10, b=10), yaxis_title="t/day",
                              legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig, use_container_width=True)

# ---- TAB 2: PRODUCTION PLAN ----
with tab2:
    st.subheader("Recommended production schedule by line")
    st.caption("Each line runs its variants in sequence, one changeover per variant (minimised). "
               "Cells show tonnes produced per day.")
    for line in PRODUCTS:
        r = results[line]
        feas = "✅ feasible" if r["feasible"] else "⚠️ capacity-constrained"
        st.markdown(f"### {line}  ·  {LINES[line]['capacity_tpd']:.0f} t/day  ·  "
                    f"utilisation {r['util']*100:.0f}%  ·  {feas}")
        st.markdown(f"**Run sequence:** {' → '.join(r['sequence']) if r['sequence'] else '—'}  "
                    f"({len(r['sequence'])} changeovers, {r['co_hours']:.1f} hr)")
        sched = pd.DataFrame({v: [round(r["plan"][v][d], 1) for d in range(horizon)]
                              for v in LINES[line]["variants"]}, index=date_labels).T
        st.dataframe(sched, use_container_width=True)
        # production heatmap
        st.plotly_chart(px.imshow(sched, text_auto=".0f", aspect="auto",
            color_continuous_scale="Blues", labels=dict(color="t")).update_layout(height=160+40*len(sched), margin=dict(t=10,b=10)),
            use_container_width=True)
        st.divider()

# ---- TAB 3: COSTS ----
with tab3:
    st.subheader("Cost performance vs reactive baseline")
    tot_opt = sum(results[l]["total"] for l in PRODUCTS)
    tot_base = sum(results[l]["baseline"] for l in PRODUCTS)
    tot_co = sum(results[l]["co_cost"] for l in PRODUCTS)
    tot_hold = sum(results[l]["hold_cost"] for l in PRODUCTS)
    tot_so = sum(results[l]["stockout_cost"] for l in PRODUCTS)
    saving = 100*(tot_base-tot_opt)/tot_base if tot_base else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Optimised total", f"${tot_opt:,.0f}")
    m2.metric("Baseline total", f"${tot_base:,.0f}")
    m3.metric("Projected saving", f"{saving:.1f}%")
    m4.metric("Total changeovers", f"{sum(len(results[l]['sequence']) for l in PRODUCTS)}")

    # per-line cost table
    rows = []
    for line in PRODUCTS:
        r = results[line]
        rows.append([line, r["co_cost"], r["hold_cost"], r["stockout_cost"], r["total"], r["baseline"]])
    cost_df = pd.DataFrame(rows, columns=["Line", "Changeover $", "Holding $", "Stockout $", "Optimised $", "Baseline $"]).set_index("Line").round(0)
    st.dataframe(cost_df, use_container_width=True)

    comp = pd.DataFrame({"Changeover": tot_co, "Holding": tot_hold, "Stockout": tot_so}, index=["Optimised"]).T
    st.plotly_chart(px.bar(comp.reset_index(), x="index", y="Optimised", color="index",
        color_discrete_sequence=["#1F4E79", "#5B9BD5", "#C00000"],
        title="Optimised cost breakdown (all lines)").update_layout(height=350, showlegend=False,
        xaxis_title="", yaxis_title="$"), use_container_width=True)

# ---- TAB 4: LINES & MODEL ----
with tab4:
    st.subheader("Production line configuration")
    cfg = []
    for line in PRODUCTS:
        for v, meta in LINES[line]["variants"].items():
            cfg.append([line, LINES[line]["capacity_tpd"], v, f"{meta['share']*100:.0f}%", meta["changeover_hr"]])
    st.dataframe(pd.DataFrame(cfg, columns=["Line", "Capacity (t/day)", "Variant", "Demand share", "Changeover (hr)"]), use_container_width=True)
    st.markdown("""
**Line capacities (calibrated from the expert interview):**
- **Maize Meal — 480 t/day:** combined milling ~20 t/hr at 97% (3% downtime) over a 24-hour, 4-shift day.
- **Flour — 300 t/day:** super-white flour line running at the full ~20 t/hr; bran flour shares the line.
- **Rice — 150 t/day:** rice is milled/packed rather than ground, modelled at ~12 t/hr packing.
- **Pasta — 60 t/day:** pasta line commissioned Feb 2024; 48 t/day nominal, 60 t/day ceiling.

**Forecasting model:** validation figures are produced by the training notebook (LSTM, XGBoost, SARIMA).
Replace the placeholder figures below with your notebook's hold-out results.
""")
    val = pd.DataFrame({"RMSE": [26.7, 15.4, 34.7], "MAE": [17.6, 10.0, 27.2], "MAPE %": [10.4, 5.6, 7.4]},
                       index=["LSTM (global)", "XGBoost", "SARIMA (maize meal)"])
    st.dataframe(val, use_container_width=True)

st.caption("Industrial engineering rules: demand-driven lot sizing · within-line changeover minimisation "
           "(one run per variant) · dedicated-line capacity balancing · inventory & service-level control.")
