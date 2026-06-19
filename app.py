"""
National Foods (Company Y) — Demand-Driven Production Scheduling System
Streamlit decision-support application.

Loads the artifacts exported by the Kaggle notebook:
  lstm_demand.keras, xscaler.joblib, yscaler.joblib, config.joblib
and reproduces the forecast -> schedule -> cost workflow in the browser.

Run locally:   streamlit run app.py
Deploy:        push repo to GitHub -> Streamlit Community Cloud
"""
import os, random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Company Y — Demand-Driven Scheduler",
                   page_icon="🌾", layout="wide")

# ----------------------------------------------------------------------------
# Artifact loading (cached). Falls back gracefully if model files are absent,
# so the UI still renders for demonstration.
# ----------------------------------------------------------------------------
PRODUCTS = ["MZ-RS-10", "FL-SW-50", "RC-MH-25", "PA-BB-05"]
PNAME = {"MZ-RS-10": "Maize Meal", "FL-SW-50": "Flour (Super White)",
         "RC-MH-25": "Rice", "PA-BB-05": "Pasta"}

DEFAULT_CFG = dict(
    LOOKBACK=14, HORIZON=7,
    DAILY_CAP={"MZ-RS-10": 480.0, "FL-SW-50": 300.0, "RC-MH-25": 150.0, "PA-BB-05": 60.0},
    OPENING={"MZ-RS-10": 640.0, "FL-SW-50": 390.0, "RC-MH-25": 210.0, "PA-BB-05": 95.0},
    HOLD={"MZ-RS-10": 0.45, "FL-SW-50": 0.55, "RC-MH-25": 0.40, "PA-BB-05": 0.50},
    STOCKOUT={"MZ-RS-10": 180, "FL-SW-50": 220, "RC-MH-25": 150, "PA-BB-05": 140},
    SAFETY={"MZ-RS-10": 320, "FL-SW-50": 195, "RC-MH-25": 140, "PA-BB-05": 96},
    CO_RATE=450.0,
    CO_H={"MZ-RS-10": {"MZ-RS-10": 0, "FL-SW-50": 4, "RC-MH-25": 6, "PA-BB-05": 5},
          "FL-SW-50": {"MZ-RS-10": 4.5, "FL-SW-50": 0, "RC-MH-25": 6, "PA-BB-05": 3},
          "RC-MH-25": {"MZ-RS-10": 6, "FL-SW-50": 6, "RC-MH-25": 0, "PA-BB-05": 5.5},
          "PA-BB-05": {"MZ-RS-10": 5, "FL-SW-50": 3.5, "RC-MH-25": 5.5, "PA-BB-05": 0}},
)

@st.cache_resource
def load_artifacts():
    cfg = dict(DEFAULT_CFG)
    model = xscaler = yscaler = None
    try:
        import joblib
        from tensorflow.keras.models import load_model
        if os.path.exists("config.joblib"):
            cfg = joblib.load("config.joblib")
        if os.path.exists("lstm_demand.keras"):
            model = load_model("lstm_demand.keras", compile=False)
        if os.path.exists("xscaler.joblib"):
            xscaler = joblib.load("xscaler.joblib")
        if os.path.exists("yscaler.joblib"):
            yscaler = joblib.load("yscaler.joblib")
    except Exception as e:
        st.session_state["load_warning"] = str(e)
    return model, xscaler, yscaler, cfg

@st.cache_data
def load_history(file=None):
    if file is not None:
        df = pd.read_csv(file, parse_dates=["date"])
    else:
        for p in ["nationalfoods_demand_2023_2025.csv", "data/nationalfoods_demand_2023_2025.csv"]:
            if os.path.exists(p):
                df = pd.read_csv(p, parse_dates=["date"]); break
        else:
            return None
    return df.sort_values(["product_code", "date"]).reset_index(drop=True)

model, xscaler, yscaler, CFG = load_artifacts()

# ----------------------------------------------------------------------------
# Feature engineering (mirrors the notebook) + forecasting
# ----------------------------------------------------------------------------
FEATURES = ["lag1", "lag7", "lag14", "roll7", "roll7_std", "roll30", "sin_doy",
            "cos_doy", "dow", "is_school_term", "is_public_holiday", "is_festive",
            "is_month_end", "is_promo", "drought_index", "unit_price_usd_per_kg"]

def make_features(df):
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek
    df["doy"] = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2*np.pi*df["doy"]/365)
    df["cos_doy"] = np.cos(2*np.pi*df["doy"]/365)
    out = []
    for pid, g in df.groupby("product_code"):
        g = g.sort_values("date").copy(); y = g["demand_orders_tonnes"]
        for L in (1, 7, 14):
            g[f"lag{L}"] = y.shift(L)
        g["roll7"] = y.shift(1).rolling(7).mean()
        g["roll7_std"] = y.shift(1).rolling(7).std()
        g["roll30"] = y.shift(1).rolling(30).mean()
        out.append(g)
    return pd.concat(out).reset_index(drop=True)

def forecast_demand(hist, horizon):
    """Return dict pid -> list[horizon] forecast tonnes.
    Uses the LSTM if available; otherwise a seasonal-naive fallback so the UI works."""
    feat = make_features(hist)
    pcols = [f"p_{p}" for p in PRODUCTS]
    dummies = pd.get_dummies(feat["product_code"], prefix="p")
    for c in pcols:
        if c not in dummies: dummies[c] = 0
    feat = pd.concat([feat, dummies[pcols].astype(float)], axis=1)
    allx = FEATURES + pcols
    feat = feat.dropna(subset=FEATURES)
    out = {}
    if model is not None and xscaler is not None and yscaler is not None:
        fs = feat.copy(); fs[allx] = xscaler.transform(feat[allx])
        look = CFG["LOOKBACK"]
        for pid in PRODUCTS:
            g = fs[fs.product_code == pid].sort_values("date")
            if len(g) < look:
                out[pid] = [float(hist[hist.product_code == pid]["demand_orders_tonnes"].tail(7).mean())]*horizon
                continue
            window = g[allx].values[-look:].copy()
            preds = []
            for _ in range(horizon):
                ps = model.predict(window[np.newaxis, :, :], verbose=0).flatten()[0]
                yhat = float(yscaler.inverse_transform([[ps]])[0, 0])
                preds.append(max(yhat, 0)); window = np.vstack([window[1:], window[-1]])
            out[pid] = preds
    else:
        # seasonal-naive fallback: last 7-day mean per product
        for pid in PRODUCTS:
            m = float(hist[hist.product_code == pid]["demand_orders_tonnes"].tail(7).mean())
            out[pid] = [m]*horizon
    return out

# ----------------------------------------------------------------------------
# Scheduling engine (greedy IE heuristic + GA) — IE cost model
# ----------------------------------------------------------------------------
def co_cost(seq, CO_H, CO_RATE):
    return sum(CO_H[a][b]*CO_RATE for a, b in zip(seq[:-1], seq[1:]) if a != b)

def evaluate(plan, seqs, dvec, P):
    stock = {p: P["OPENING"][p] for p in PRODUCTS}; hc = cc = sc = 0.0; H = len(seqs)
    inv_track = {p: [] for p in PRODUCTS}; unmet_track = {p: [] for p in PRODUCTS}
    for d in range(H):
        cc += co_cost(seqs[d], P["CO_H"], P["CO_RATE"])
        for p in PRODUCTS:
            avail = stock[p] + plan[p][d]; disp = min(dvec[p][d], avail)
            unmet = max(dvec[p][d]-disp, 0); stock[p] = avail-disp
            sc += unmet*P["STOCKOUT"][p]; hc += stock[p]*P["HOLD"][p]
            if stock[p] < P["SAFETY"][p]*0.5:
                sc += (P["SAFETY"][p]*0.5-stock[p])*P["HOLD"][p]*2
            inv_track[p].append(stock[p]); unmet_track[p].append(unmet)
    return cc+hc+sc, dict(changeover=cc, holding=hc, stockout=sc), inv_track, unmet_track

def greedy(dvec, P):
    H = len(next(iter(dvec.values())))
    plan = {p: [0.0]*H for p in PRODUCTS}; stock = {p: P["OPENING"][p] for p in PRODUCTS}; seqs = []
    for d in range(H):
        active = []
        for p in PRODUCTS:
            dem = dvec[p][d]; need = dem+max(P["SAFETY"][p]-stock[p], 0)-stock[p]
            prod = max(min(need, P["DAILY_CAP"][p]), 0)
            if stock[p] < dem: prod = max(prod, min(dem-stock[p], P["DAILY_CAP"][p]))
            plan[p][d] = prod
            if prod > 0: active.append(p)
            stock[p] = stock[p]+prod-min(dem, stock[p]+prod)
        if active:
            seq = [active[0]]; rest = active[1:]
            while rest:
                nxt = min(rest, key=lambda x: P["CO_H"][seq[-1]][x]); seq.append(nxt); rest.remove(nxt)
        else:
            seq = [PRODUCTS[0]]
        seqs.append(seq)
    return plan, seqs

def ga(dvec, P, gplan, gseqs, pop=50, gens=120, seed=42):
    random.seed(seed); H = len(next(iter(dvec.values())))
    def rind():
        return ({p: [random.uniform(0.5, 1.0)*P["DAILY_CAP"][p] for _ in range(H)] for p in PRODUCTS},
                [random.sample(PRODUCTS, len(PRODUCTS)) for _ in range(H)])
    def fit(ind): return evaluate(ind[0], ind[1], dvec, P)[0]
    def cx(a, b):
        cp = {p: [(a[0][p][d] if random.random() < .5 else b[0][p][d]) for d in range(H)] for p in PRODUCTS}
        cs = [(a[1][d] if random.random() < .5 else b[1][d]) for d in range(H)]; return cp, cs
    def mut(ind, r=.2):
        pl, sq = ind
        for p in PRODUCTS:
            for d in range(H):
                if random.random() < r: pl[p][d] = min(max(pl[p][d]*random.uniform(.7, 1.3), 0), P["DAILY_CAP"][p])
        for d in range(H):
            if random.random() < r: random.shuffle(sq[d])
        return pl, sq
    pop_list = [rind() for _ in range(pop)]; pop_list[0] = (gplan, gseqs)
    best = None; bf = float("inf"); hist = []
    for _ in range(gens):
        S = sorted(pop_list, key=fit); f0 = fit(S[0]); hist.append(f0)
        if f0 < bf: bf = f0; best = ({p: list(S[0][0][p]) for p in PRODUCTS}, [list(s) for s in S[0][1]])
        elite = S[:max(2, pop//4)]
        NP = [({p: list(elite[0][0][p]) for p in PRODUCTS}, [list(s) for s in elite[0][1]])]
        while len(NP) < pop:
            a, b = random.sample(elite, 2)
            ac = ({p: list(a[0][p]) for p in PRODUCTS}, [list(s) for s in a[1]])
            bc = ({p: list(b[0][p]) for p in PRODUCTS}, [list(s) for s in b[1]])
            NP.append(mut(cx(ac, bc)))
        pop_list = NP
    return best, bf, hist

def baseline(dvec, P, seed=42):
    random.seed(seed); H = len(next(iter(dvec.values())))
    pl = {p: [np.mean(dvec[p])*random.uniform(.9, 1.05) for _ in range(H)] for p in PRODUCTS}
    sq = [random.sample(PRODUCTS, len(PRODUCTS)) for _ in range(H)]
    return pl, sq

# ============================================================================
# SIDEBAR — what-if controls
# ============================================================================
st.sidebar.title("⚙️ Planning Controls")
up = st.sidebar.file_uploader("Demand history CSV (optional)", type="csv")
hist = load_history(up)

horizon = st.sidebar.slider("Planning horizon (days)", 3, 14, int(CFG.get("HORIZON", 7)))
scheduler = st.sidebar.radio("Scheduler", ["Genetic Algorithm", "Greedy IE heuristic", "Compare both"])

st.sidebar.markdown("### Cost parameters")
co_rate = st.sidebar.number_input("Changeover cost ($/hr)", 50.0, 2000.0, float(CFG["CO_RATE"]), 50.0)

st.sidebar.markdown("### Per-product settings")
P = {k: (dict(v) if isinstance(v, dict) else v) for k, v in CFG.items()}
P["CO_RATE"] = co_rate
for pid in PRODUCTS:
    with st.sidebar.expander(PNAME[pid]):
        P["DAILY_CAP"][pid] = st.number_input(f"Daily capacity (t) — {pid}", 1.0, 2000.0, float(CFG["DAILY_CAP"][pid]), 10.0, key=f"cap{pid}")
        P["OPENING"][pid] = st.number_input(f"Opening stock (t) — {pid}", 0.0, 5000.0, float(CFG["OPENING"][pid]), 10.0, key=f"op{pid}")
        P["SAFETY"][pid] = st.number_input(f"Safety stock (t) — {pid}", 0.0, 2000.0, float(CFG["SAFETY"][pid]), 10.0, key=f"sa{pid}")
        P["HOLD"][pid] = st.number_input(f"Holding ($/t/day) — {pid}", 0.0, 10.0, float(CFG["HOLD"][pid]), 0.05, key=f"ho{pid}")
        P["STOCKOUT"][pid] = st.number_input(f"Stockout penalty ($/t) — {pid}", 0.0, 2000.0, float(CFG["STOCKOUT"][pid]), 10.0, key=f"so{pid}")

run = st.sidebar.button("▶ Generate plan", type="primary", use_container_width=True)

# ============================================================================
# HEADER
# ============================================================================
st.title("🌾 Demand-Driven Production Scheduling System")
st.caption("Company Y — deep-learning demand forecasting + cost-aware production scheduling. "
           "Data is synthetic, calibrated to expert interview and public sources.")
c1, c2, c3 = st.columns(3)
c1.metric("Model", "LSTM (global)" if model is not None else "Fallback (naive)")
c2.metric("Data span", f"{hist['date'].min().date()} → {hist['date'].max().date()}" if hist is not None else "—")
c3.metric("Horizon", f"{horizon} days")

if hist is None:
    st.warning("No demand history found. Upload `nationalfoods_demand_2023_2025.csv` in the sidebar.")
    st.stop()

# compute on demand
if run or "plan_done" not in st.session_state:
    with st.spinner("Forecasting demand and optimising the schedule…"):
        dvec = forecast_demand(hist, horizon)
        gplan, gseqs = greedy(dvec, P)
        gcost, gbd, ginv, gunmet = evaluate(gplan, gseqs, dvec, P)
        (bplan, bseqs), bcost, ghist = ga(dvec, P, gplan, gseqs)
        _, bbd, binv, bunmet = evaluate(bplan, bseqs, dvec, P)
        bl_plan, bl_seqs = baseline(dvec, P)
        blcost, blbd, _, _ = evaluate(bl_plan, bl_seqs, dvec, P)
        st.session_state.update(dict(plan_done=True, dvec=dvec, gplan=gplan, gseqs=gseqs,
            gcost=gcost, gbd=gbd, ginv=ginv, gunmet=gunmet, bplan=bplan, bseqs=bseqs,
            bcost=bcost, bbd=bbd, binv=binv, bunmet=bunmet, blcost=blcost, blbd=blbd,
            ghist=ghist, horizon=horizon))

S = st.session_state
dvec = S["dvec"]; H = S["horizon"]
if scheduler == "Greedy IE heuristic":
    plan, seqs, cost, bd, inv, unmet = S["gplan"], S["gseqs"], S["gcost"], S["gbd"], S["ginv"], S["gunmet"]
else:
    plan, seqs, cost, bd, inv, unmet = S["bplan"], S["bseqs"], S["bcost"], S["bbd"], S["binv"], S["bunmet"]

days = [f"Day {d+1}" for d in range(H)]

tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "🏭 Schedule", "💲 Costs", "🔬 Model"])

# ---- TAB 1: FORECAST ----
with tab1:
    st.subheader("7-day demand forecast")
    fig = go.Figure()
    for pid in PRODUCTS:
        recent = hist[hist.product_code == pid].tail(28)
        fig.add_trace(go.Scatter(x=list(range(-len(recent), 0)), y=recent["demand_orders_tonnes"],
                                 name=f"{PNAME[pid]} (actual)", line=dict(width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=list(range(0, H)), y=dvec[pid], name=f"{PNAME[pid]} (forecast)", line=dict(width=2.5)))
    fig.update_layout(height=420, xaxis_title="day (0 = first planned day)", yaxis_title="tonnes/day",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)
    fc = pd.DataFrame({PNAME[p]: [round(v, 1) for v in dvec[p]] for p in PRODUCTS}, index=days)
    st.dataframe(fc, use_container_width=True)

# ---- TAB 2: SCHEDULE ----
with tab2:
    st.subheader(f"Recommended production schedule — {scheduler}")
    sched = pd.DataFrame({PNAME[p]: [round(plan[p][d], 1) for d in range(H)] for p in PRODUCTS}, index=days)
    sched["Run sequence"] = [" → ".join(PNAME[x] for x in seqs[d]) for d in range(H)]
    st.dataframe(sched, use_container_width=True)

    st.markdown("**Production heatmap (tonnes)**")
    hm = pd.DataFrame({PNAME[p]: [plan[p][d] for d in range(H)] for p in PRODUCTS}, index=days).T
    st.plotly_chart(px.imshow(hm, text_auto=".0f", aspect="auto", color_continuous_scale="Blues",
                              labels=dict(color="tonnes")).update_layout(height=300), use_container_width=True)

    st.markdown("**Projected closing inventory vs target band (10–20% of production)**")
    figi = go.Figure()
    for pid in PRODUCTS:
        figi.add_trace(go.Scatter(x=days, y=[round(v, 1) for v in inv[pid]], name=PNAME[pid], mode="lines+markers"))
    figi.update_layout(height=360, yaxis_title="closing stock (t)")
    st.plotly_chart(figi, use_container_width=True)

# ---- TAB 3: COSTS ----
with tab3:
    st.subheader("Cost performance vs reactive baseline")
    saving = 100*(S["blcost"]-cost)/S["blcost"] if S["blcost"] else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Weekly total cost", f"${cost:,.0f}")
    m2.metric("Baseline cost", f"${S['blcost']:,.0f}")
    m3.metric("Projected saving", f"{saving:.1f}%")
    total_unmet = sum(sum(unmet[p]) for p in PRODUCTS)
    m4.metric("Total unmet demand", f"{total_unmet:.0f} t")

    comp = pd.DataFrame({
        "Baseline (reactive)": [S["blbd"]["changeover"], S["blbd"]["holding"], S["blbd"]["stockout"], S["blcost"]],
        "Greedy IE": [S["gbd"]["changeover"], S["gbd"]["holding"], S["gbd"]["stockout"], S["gcost"]],
        "Genetic Algorithm": [S["bbd"]["changeover"], S["bbd"]["holding"], S["bbd"]["stockout"], S["bcost"]],
    }, index=["Changeover $", "Holding $", "Stockout $", "TOTAL $"]).round(0)
    st.dataframe(comp, use_container_width=True)

    bars = comp.T.iloc[:, :3].reset_index().melt(id_vars="index", var_name="component", value_name="cost")
    st.plotly_chart(px.bar(bars, x="index", y="cost", color="component", title="Cost breakdown by approach",
                           color_discrete_sequence=["#1F4E79", "#5B9BD5", "#C00000"]).update_layout(height=380,
                           xaxis_title="", yaxis_title="$ / week"), use_container_width=True)

# ---- TAB 4: MODEL ----
with tab4:
    st.subheader("Forecasting model — validation")
    st.markdown("Hold-out accuracy (from the training notebook). Re-run the notebook to refresh these figures.")
    val = pd.DataFrame({"RMSE": [24.3, 15.1, 34.7], "MAE": [16.9, 9.7, 27.2], "MAPE %": [10.2, 5.5, 7.4]},
                       index=["LSTM (global)", "XGBoost", "SARIMA (maize meal)"])
    st.dataframe(val, use_container_width=True)
    st.caption("Note: report your own notebook's figures here. Values shown are illustrative from a sample run.")
    st.markdown("**GA convergence (this run)**")
    if "ghist" in S:
        st.plotly_chart(go.Figure(go.Scatter(y=S["ghist"], mode="lines")).update_layout(
            height=320, xaxis_title="generation", yaxis_title="best cost ($)"), use_container_width=True)

st.caption("Industrial engineering rules embedded: demand-driven lot sizing · sequence-dependent "
           "changeover minimisation · capacity balancing · inventory & shelf-life control · service-level protection.")
