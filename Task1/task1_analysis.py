#!/usr/bin/env python3
"""
Task 1 - Cyclone machine data analysis (end-to-end, runnable).

Usage:  python task1_analysis.py [path/to/data.xlsx]
Outputs (written next to this script): CSVs, plots/*.png, insights.md
Requires: pandas, numpy, scikit-learn, matplotlib, scipy, openpyxl
"""
import sys, os, json, time, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots"); os.makedirs(PLOTS, exist_ok=True)
def find_data():
    if len(sys.argv) > 1: return sys.argv[1]
    for p in (os.path.join(HERE, "data.xlsx"), os.path.join(HERE, "..", "data.xlsx"), os.path.join(os.getcwd(), "data.xlsx")):
        if os.path.exists(p): return p
    sys.exit("data.xlsx not found - pass its path:  python task1_analysis.py /path/to/data.xlsx")
DATA = find_data()
SEED = 42; np.random.seed(SEED)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
def save(name): plt.tight_layout(); plt.savefig(os.path.join(PLOTS, name), dpi=130); plt.close()

VARS = ["Cyclone_Inlet_Gas_Temp", "Cyclone_Gas_Outlet_Temp", "Cyclone_Material_Temp",
        "Cyclone_Inlet_Draft", "Cyclone_Outlet_Gas_draft", "Cyclone_cone_draft"]
TEMP, DRAFT = "Cyclone_Inlet_Gas_Temp", "Cyclone_Inlet_Draft"
STEP = pd.Timedelta("5min")
INSIGHT = {}   # numbers reused in insights.md / presentation

# ------------------------------------------------------------------ 1. LOAD + CLEAN
log("loading data")
cache = os.path.join(HERE, ".raw_cache.pkl")
if os.path.exists(cache): raw = pd.read_pickle(cache)
else:
    raw = pd.read_excel(DATA); raw.to_pickle(cache)
raw = raw.rename(columns={"time": "timestamp"})
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
raw = raw.sort_values("timestamp").drop_duplicates("timestamp")

# 1a. text error codes ("I/O Timeout", "Not Connect", ...) -> NaN
err_counts = {}; DQ = {c: {} for c in VARS}
for c in VARS:
    num = pd.to_numeric(raw[c], errors="coerce")
    bad = raw[c][num.isna() & raw[c].notna()].astype(str)
    for k, v in bad.value_counts().items(): err_counts[k] = err_counts.get(k, 0) + int(v)
    DQ[c]["text_error_cells"] = int((num.isna() & raw[c].notna()).sum()); raw[c] = num
INSIGHT["text_error_cells"] = int(sum(err_counts.values())); INSIGHT["text_error_types"] = err_counts

# 1b. strict 5-minute index (missing timestamps become NaN rows)
n_raw = len(raw)
df = raw.set_index("timestamp")[VARS].asfreq("5min")
INSIGHT.update(rows_raw=n_raw, rows_indexed=len(df), missing_timestamps=len(df) - n_raw,
               start=str(df.index[0]), end=str(df.index[-1]))
log(f"{n_raw} raw rows -> {len(df)} on strict 5-min grid ({len(df)-n_raw} missing timestamps)")

# 1c. sensor-range sentinels: values pinned at the instrument limits are not real readings
n_sent = 0
for c, bad in {"Cyclone_Material_Temp": [-185.0, 1375.0], "Cyclone_Gas_Outlet_Temp": [1375.0]}.items():
    m = df[c].isin(bad); n_sent += int(m.sum()); DQ[c]["sentinel_values"] = int(m.sum()); df.loc[m, c] = np.nan
# Material temp reads exactly 0.0 when its thermocouple is down while the line is hot
m = (df["Cyclone_Material_Temp"] == 0) & (df[TEMP] > 400); n_sent += int(m.sum()); DQ["Cyclone_Material_Temp"]["sentinel_values"] = DQ["Cyclone_Material_Temp"].get("sentinel_values", 0) + int(m.sum())
df.loc[m, "Cyclone_Material_Temp"] = np.nan
INSIGHT["sentinel_values_nulled"] = n_sent

# 1d. spike (outlier) removal: Hampel filter on 1-step jumps relative to local median, very conservative
clean = df.copy(); n_out = 0
for c in VARS:
    med = df[c].rolling(13, center=True, min_periods=5).median()
    dev = (df[c] - med).abs()
    mad = dev.rolling(289, center=True, min_periods=50).median().clip(lower=1e-3) * 1.4826
    # a point is a spike if far from local median AND both neighbours are back near it (isolated)
    isolated = ((df[c].shift(1) - med).abs() < 4 * mad) & ((df[c].shift(-1) - med).abs() < 4 * mad)
    spike = (dev > 10 * mad) & (dev > 0.15 * df[c].abs().rolling(289, center=True, min_periods=50).median()) & isolated
    n_out += int(spike.sum()); DQ[c]["isolated_spikes_removed"] = int(spike.sum()); clean.loc[spike, c] = np.nan
INSIGHT["isolated_spikes_removed"] = n_out

# 1e. short gaps (<= 30 min) linearly interpolated; longer gaps stay NaN (flagged, excluded from modelling)
nan_before = int(clean.isna().any(axis=1).sum())
clean = clean.interpolate(limit=6, limit_area="inside")
INSIGHT["rows_with_nan_before_interp"] = nan_before
INSIGHT["rows_with_nan_after_interp"] = int(clean.isna().any(axis=1).sum())
data_gap = clean[VARS].isna().any(axis=1)
dq = pd.DataFrame(DQ).T.fillna(0).astype(int)
dq["missing_timestamps_rows"] = len(df) - n_raw
dq["nan_before_interpolation"] = df[VARS].isna().sum() + 0
dq["nan_left_after_interpolation"] = clean[VARS].isna().sum()
dq.index.name = "variable"; dq.to_csv(os.path.join(HERE, "data_quality_report.csv"))
log(f"cleaned: {n_out} spikes, {n_sent} sentinels, NaN rows {nan_before} -> {int(data_gap.sum())}")

# ------------------------------------------------------------------ 2. EDA
stats = clean[VARS].describe(percentiles=[.01, .05, .5, .95, .99]).T
stats.to_csv(os.path.join(HERE, "summary_statistics.csv"))
corr = clean[VARS].corr()
corr.to_csv(os.path.join(HERE, "correlation_matrix.csv"))
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
short = [v.replace("Cyclone_", "") for v in VARS]
ax.set_xticks(range(6)); ax.set_xticklabels(short, rotation=45, ha="right"); ax.set_yticks(range(6)); ax.set_yticklabels(short)
for i in range(6):
    for j in range(6): ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar(im); ax.set_title("Correlation matrix (cleaned data, all states)"); save("01_correlation_matrix.png")

def panel(sl, title, fname, fill=None):
    fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    d = clean.loc[sl]
    axs[0].plot(d.index, d[TEMP], lw=.6, label="Inlet gas T"); axs[0].plot(d.index, d["Cyclone_Gas_Outlet_Temp"], lw=.6, label="Outlet gas T")
    axs[0].plot(d.index, d["Cyclone_Material_Temp"], lw=.6, label="Material T"); axs[0].set_ylabel("Temp"); axs[0].legend(ncol=3, loc="lower left")
    for c in ["Cyclone_Inlet_Draft", "Cyclone_Outlet_Gas_draft", "Cyclone_cone_draft"]: axs[1].plot(d.index, d[c], lw=.6, label=c.replace("Cyclone_", ""))
    axs[1].set_ylabel("Draft"); axs[1].legend(ncol=3, loc="lower left")
    axs[2].plot(d.index, d[TEMP].rolling(12).std(), lw=.6, color="gray"); axs[2].set_ylabel("Inlet T rolling std (1h)")
    axs[0].set_title(title); save(fname)
# representative normal week + a week containing a shutdown restart are chosen after shutdown detection
# ------------------------------------------------------------------ 3. SHUTDOWN DETECTION
log("detecting shutdowns")
# Physical signature: when the ID fan is off / line is tripped, ALL drafts collapse to ~0 (normal ~ -150..-250)
# and temperatures decay towards ambient. Draft is the cleanest, fastest indicator (temps lag).
draft_med = clean[["Cyclone_Inlet_Draft", "Cyclone_Outlet_Gas_draft", "Cyclone_cone_draft"]].median(axis=1)
down = (draft_med > -60)
down = down.where(~draft_med.isna())                      # NaN where unknown
# fill unknown stretches only if the state is identical on both sides of the gap
prev = down.ffill(); nxt = down.bfill()
down = down.fillna(prev.where(prev == nxt)).fillna(False).astype(bool)
MIN_DOWN, MERGE_GAP = 6, 6      # >= 30 min to count as a shutdown; merge stops separated by <= 30 min of run
def segments(mask):
    r = (mask != mask.shift()).cumsum()
    g = mask.groupby(r); s = g.apply(lambda x: x.index[0]); e = g.apply(lambda x: x.index[-1]); v = g.first()
    return [(a, b) for a, b, f in zip(s, e, v) if f]
segs = segments(down)
merged = []
for a, b in segs:
    if merged and (a - merged[-1][1]) / STEP <= MERGE_GAP + 1: merged[-1] = (merged[-1][0], b)
    else: merged.append((a, b))
sd = [(a, b) for a, b in merged if (b - a) / STEP + 1 >= MIN_DOWN]
shut = pd.DataFrame(sd, columns=["start", "end"])
shut["end_exclusive"] = shut["end"] + STEP                      # first timestamp back in operation
shut["duration_min"] = ((shut["end"] - shut["start"]) / STEP + 1) * 5
shut["duration_hours"] = (shut["duration_min"] / 60).round(2)
shut["min_inlet_temp"] = [clean.loc[a:b, TEMP].min() for a, b in sd]
shut["type"] = pd.cut(shut["duration_hours"], [0, 2, 12, 48, 1e9], labels=["short_stop(<2h)", "idle(2-12h)", "shutdown(12-48h)", "extended_outage(>48h)"])
shut[["start", "end", "duration_min", "duration_hours", "min_inlet_temp", "type"]].to_csv(os.path.join(HERE, "shutdown_periods.csv"), index=False)
is_down = pd.Series(False, index=clean.index)
for a, b in sd: is_down.loc[a:b] = True
tot_h = shut["duration_hours"].sum(); span_h = len(clean) * 5 / 60
INSIGHT.update(n_shutdowns=len(shut), total_downtime_h=round(tot_h, 1), total_downtime_days=round(tot_h/24, 1),
               downtime_pct=round(100*is_down.mean(), 1), median_shutdown_h=float(shut.duration_hours.median()),
               longest_shutdown_h=float(shut.duration_hours.max()), types=shut["type"].value_counts().to_dict())
log(f"{len(shut)} shutdown events, {tot_h:.0f} h total ({100*is_down.mean():.1f}% of time)")

# yearly view with shutdowns highlighted
YEAR = 2018
fig, ax = plt.subplots(figsize=(15, 5))
y = clean.loc[str(YEAR)]
ax.plot(y.index, y[TEMP], lw=.5, color="tab:blue", label="Inlet gas temp")
ax.plot(y.index, y["Cyclone_Material_Temp"], lw=.4, color="tab:orange", alpha=.6, label="Material temp")
for i, (a, b) in enumerate(sd):
    if a.year == YEAR or b.year == YEAR: ax.axvspan(a, b + STEP, color="red", alpha=.25, label="Shutdown/idle" if i == 0 or not ax.get_legend_handles_labels()[1].count("Shutdown/idle") else None)
ax.set_title(f"{YEAR}: Inlet gas temperature with detected shutdown / idle periods (red)"); ax.set_ylabel("Temperature"); ax.legend(loc="lower left")
save("04_shutdowns_year_2018.png")
fig, axs = plt.subplots(1, 2, figsize=(13, 4))
axs[0].hist(shut.duration_hours.clip(upper=200), bins=40, color="tab:red"); axs[0].set_xlabel("Shutdown duration (h, clipped at 200)"); axs[0].set_title("Shutdown duration distribution")
mon = shut.groupby(shut.start.dt.to_period("M")).duration_hours.sum().reindex(pd.period_range(clean.index[0], clean.index[-1], freq="M"), fill_value=0)
axs[1].bar(range(len(mon)), mon.values, color="tab:red"); axs[1].set_xticks(range(0, len(mon), 6)); axs[1].set_xticklabels([str(p) for p in mon.index[::6]], rotation=45)
axs[1].set_title("Downtime hours per month"); save("05_shutdown_summary.png")

# the shutdown rule rests on a gap in the distribution of the median draft: an empty valley around -60
fig, ax = plt.subplots(figsize=(9, 4)); dm = draft_med.dropna().clip(-350, 50)
ax.hist(dm, bins=140, color="steelblue"); ax.axvline(-60, color="k", ls="--", label="threshold (-60)"); ax.set_yscale("log")
ax.set_xlabel("median of the three drafts (clipped)"); ax.set_ylabel("5-min samples (log)"); ax.set_title("Median of the three drafts: running vs fan-off, with a near-empty valley between", fontsize=11); ax.legend()
save("05b_median_draft_histogram.png")
hot = [(clean.loc[a:b, TEMP] > 700).mean() for a, b in sd]
INSIGHT["shutdowns_inlet_above_700C_most_of_length"] = int(sum(h_ > 0.5 for h_ in hot))
INSIGHT["samples_between_-100_and_-40_pct"] = round(100 * float(((draft_med > -100) & (draft_med < -40)).sum() / draft_med.notna().sum()), 2)
INSIGHT["shutdown_type_counts"] = shut["type"].value_counts().to_dict()
INSIGHT["longest_shutdown_days"] = round(float(shut.duration_hours.max() / 24), 1)

# representative slices: one normal week and one week around a restart; plus 1-year overview
act_week = None
for st in pd.date_range(clean.index[0], clean.index[-1] - pd.Timedelta("7D"), freq="7D"):
    if (~is_down[st:st + pd.Timedelta("7D")]).all() and not data_gap[st:st + pd.Timedelta("7D")].any(): act_week = st; break
panel(slice(act_week, act_week + pd.Timedelta("7D")), f"Normal operation: one week from {act_week.date()}", "02_week_normal.png")
mid = shut.loc[shut.duration_hours.sub(24).abs().idxmin(), "start"]
panel(slice(mid - pd.Timedelta("3D"), mid + pd.Timedelta("4D")), f"Week containing a shutdown (start {mid})", "03_week_with_shutdown.png")

# one full year of normal behaviour + variance (daily mean with 5-95 % band), downtime shaded
fig, axs = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
yy = clean.loc["2018"]
for ax, cols_, ttl in ((axs[0], [TEMP, "Cyclone_Gas_Outlet_Temp", "Cyclone_Material_Temp"], "Temperatures"),
                       (axs[1], ["Cyclone_Inlet_Draft", "Cyclone_Outlet_Gas_draft", "Cyclone_cone_draft"], "Drafts")):
    for v in cols_:
        g = yy[v].resample("1D"); ax.plot(g.mean().index, g.mean(), lw=1, label=v.replace("Cyclone_", ""))
        ax.fill_between(g.mean().index, g.quantile(.05), g.quantile(.95), alpha=.12)
    for a, b in sd:
        if a.year == 2018 or b.year == 2018: ax.axvspan(a, b + STEP, color="red", alpha=.12)
    ax.set_ylabel(ttl); ax.legend(ncol=3, loc="lower left", fontsize=8)
axs[0].set_title("2018 overview: daily mean and 5-95 % band per sensor (red = shutdown / idle)")
save("02b_year_overview_2018.png")

# ------------------------------------------------------------------ 4. CLUSTERING (active only)
log("clustering")
F = clean[VARS].copy()
F["dT_in_15m"] = clean[TEMP].diff(3); F["dDraft_15m"] = clean[DRAFT].diff(3)
F["std_T_1h"] = clean[TEMP].rolling(12, min_periods=8).std(); F["std_draft_1h"] = clean[DRAFT].rolling(12, min_periods=8).std()
F["T_in_minus_out"] = clean[TEMP] - clean["Cyclone_Gas_Outlet_Temp"]; F["Tmat_minus_in"] = clean["Cyclone_Material_Temp"] - clean[TEMP]
# Clustering features: 30-min TRAILING smoothing (past data only) removes 5-min sensor jitter so states persist; plus 1-h trend and volatility
S = clean[VARS].rolling(7, min_periods=4).mean()
G = pd.DataFrame({"T_in": S[TEMP], "T_out": S["Cyclone_Gas_Outlet_Temp"], "T_mat": S["Cyclone_Material_Temp"],
                  "draft_in": S["Cyclone_Inlet_Draft"], "draft_cone": S["Cyclone_cone_draft"],
                  "slope_T_1h": S[TEMP].diff(12), "std_T_1h": F["std_T_1h"], "std_draft_1h": F["std_draft_1h"]})
FEATS = list(G.columns)
active = (~is_down) & F.notna().all(axis=1) & G.notna().all(axis=1)
Xa = F[active]; Ga = G[active]
scaler = StandardScaler().fit(Ga)
Z = np.clip(scaler.transform(Ga), -5, 5)
sub = np.random.RandomState(SEED).choice(len(Z), 12000, replace=False)
sil = {}
for k in range(3, 8):
    km = KMeans(k, n_init=4, random_state=SEED).fit(Z[sub]); sil[k] = silhouette_score(Z[sub], km.labels_)
log(f"silhouette by k: {json.dumps({k: round(v,3) for k,v in sil.items()})}")
K = int(os.environ.get("K", 5))   # k=5: lowest k that isolates the unstable and heat-loss regimes (see README)
km = KMeans(K, n_init=10, random_state=SEED).fit(Z)
raw_lab = pd.Series(km.labels_, index=Ga.index)
ari_seed = [adjusted_rand_score(km.labels_, KMeans(K, n_init=3, random_state=sd_).fit(Z).labels_) for sd_ in (1, 2, 3, 4, 5)]
hA = len(Z) // 2
mA = KMeans(K, n_init=5, random_state=SEED).fit(Z[:hA]); mB = KMeans(K, n_init=5, random_state=SEED).fit(Z[hA:])
INSIGHT["k_stability"] = dict(ari_other_seeds_mean=round(float(np.mean(ari_seed)), 3), ari_other_seeds_min=round(float(np.min(ari_seed)), 3),
    ari_first_half_model_vs_full_model=round(adjusted_rand_score(km.labels_, mA.predict(Z)), 3),
    ari_second_half_model_vs_full_model=round(adjusted_rand_score(km.labels_, mB.predict(Z)), 3),
    ari_first_vs_second_half_models_on_second_half=round(adjusted_rand_score(mA.predict(Z[hA:]), mB.labels_), 3))
# temporal smoothing: majority label over a TRAILING 35-min window (past data only; removes boundary chatter)
oh = pd.get_dummies(raw_lab.reindex(clean.index)).astype(float).rolling(7, min_periods=1).mean()
lab = pd.Series(oh.values.argmax(1), index=clean.index).loc[Ga.index]; lab = pd.Series(oh.columns.values[lab.values], index=Ga.index).astype(int)
INSIGHT["label_changes_after_smoothing_pct"] = round(100 * float((lab != raw_lab).mean()), 2)
INSIGHT["silhouette"] = {k: round(v, 3) for k, v in sil.items()}; INSIGHT["K"] = K
fig, ax = plt.subplots(figsize=(5, 3.5)); ax.plot(list(sil), list(sil.values()), "o-"); ax.axvline(K, color="r", ls="--"); ax.set_xlabel("k"); ax.set_ylabel("silhouette (12k sample)"); ax.set_title("KMeans k selection"); save("06_kmeans_k_selection.png")

# data-driven, rule-based naming from centroids (ranks, not hard-coded cluster ids)
cen = Ga.groupby(lab).mean(); names = {}
unstable = cen["std_T_1h"].idxmax(); names[unstable] = "Unstable / Upset (ramp-up, high volatility)"
gap = (cen["T_in"] - cen["T_out"]).drop(unstable).idxmax(); names[gap] = "Degraded heat recovery (low outlet T)"
rest = [c for c in cen.index if c not in names]
ranked = sorted(rest, key=lambda c: cen.loc[c, "T_mat"])
labels_rest = {1: ["Normal"], 2: ["Low-load (cool)", "High-load (hot)"], 3: ["Low-load (cool)", "Normal", "High-load (hot)"]}
if len(rest) in labels_rest: names.update(dict(zip(ranked, labels_rest[len(rest)])))
else:
    for i, c in enumerate(ranked): names[c] = f"Normal-{i+1}"
state = lab.map(names)
state_full = pd.Series("SHUTDOWN", index=clean.index); state_full[data_gap & ~is_down] = "DATA_GAP"; state_full[state.index] = state
log(f"state names: {names}")

rows = []; dur_rows = []
for c, nm in names.items():
    m = lab == c; sub_x = Xa[m]
    r = {"cluster_id": c, "state": nm, "n_points": int(m.sum()), "pct_of_active_time": round(100*m.mean(), 2)}
    for v in VARS + ["dT_in_15m", "std_T_1h"]:
        s = sub_x[v]; r.update({f"{v}__mean": s.mean(), f"{v}__std": s.std(), f"{v}__p05": s.quantile(.05), f"{v}__p50": s.median(), f"{v}__p95": s.quantile(.95)})
    # episodes = consecutive 5-min points in same state (a gap <= 15 min does not break an episode)
    s_ser = state_full == nm
    rr = (s_ser != s_ser.shift()).cumsum(); lens = s_ser.groupby(rr).agg(["first", "size"]); lens = lens[lens["first"]]["size"] * 5
    r.update({"n_episodes": len(lens), "episodes_per_month": round(len(lens) / (len(clean) * 5 / 60 / 24 / 30.4), 1),
              "episode_mean_min": lens.mean(), "episode_median_min": lens.median(), "episode_p90_min": lens.quantile(.9), "episode_max_min": lens.max()})
    rows.append(r)
cs = pd.DataFrame(rows).sort_values("cluster_id").round(2)
desc = {}
gm = Xa[TEMP].median()
for _, r in cs.iterrows():
    t, d, dt = r[f"{TEMP}__mean"], r[f"{DRAFT}__mean"], r["dT_in_15m__mean"]
    desc[r.state] = (f"Inlet T mean {t:.0f}, inlet draft {d:.0f}, 15-min dT {dt:+.1f}, 1-h T std {r['std_T_1h__mean']:.1f}; "
                     f"{r.pct_of_active_time:.1f}% of active time in {int(r.n_episodes)} episodes (median {r.episode_median_min:.0f} min)")
cs["description"] = cs.state.map(desc)
cs.to_csv(os.path.join(HERE, "clusters_summary.csv"), index=False)

# profile + timeline plots
fig, ax = plt.subplots(figsize=(11, 5)); zc = (cen - Ga.mean()) / Ga.std(); zc.index = [names[c] for c in zc.index]
im = ax.imshow(zc.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto"); ax.set_yticks(range(len(zc))); ax.set_yticklabels(zc.index)
ax.set_xticks(range(len(FEATS))); ax.set_xticklabels(FEATS, rotation=45, ha="right"); plt.colorbar(im, label="z-score of cluster mean")
ax.set_title("Cluster centroid profile (standardised features)"); save("07_cluster_profile.png")
pal = {n: c for n, c in zip(sorted(set(names.values())), plt.cm.tab10.colors)}; pal["SHUTDOWN"] = (0.85, 0.1, 0.1); pal["DATA_GAP"] = (0.6, 0.6, 0.6)
fig, axs = plt.subplots(2, 1, figsize=(15, 6), sharex=True, gridspec_kw={"height_ratios": [3, .5]})
w = slice(act_week, act_week + pd.Timedelta("14D")); wd = clean.loc[w]; ws = state_full.loc[w]
axs[0].scatter(wd.index, wd[TEMP], c=[pal[s] for s in ws], s=3)
axs[1].scatter(ws.index, [0]*len(ws), c=[pal[s] for s in ws], marker="|", s=200); axs[1].set_yticks([])
axs[0].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=n) for n, c in pal.items() if n in set(ws)], ncol=4, loc="lower left")
axs[0].set_title("Inlet gas temp coloured by machine state (two weeks)"); save("08_cluster_timeline.png")
fig, ax = plt.subplots(figsize=(9, 4)); cs.set_index("state")["pct_of_active_time"].plot.barh(ax=ax, color="steelblue"); ax.set_xlabel("% of active time"); ax.set_title("State occupancy"); save("09_state_occupancy.png")

# ------------------------------------------------------------------ 5. CONTEXTUAL ANOMALY DETECTION
log("anomaly detection")
AF = VARS + ["dT_in_15m", "dDraft_15m", "std_T_1h", "T_in_minus_out", "Tmat_minus_in"]
score = pd.Series(np.nan, index=Xa.index); flag = pd.Series(False, index=Xa.index)
rz = pd.DataFrame(np.nan, index=Xa.index, columns=AF)          # state-conditional robust z-scores (implicated vars)
CONTAM = 0.01
for c, nm in names.items():
    idx = lab.index[lab == c]; X = F.loc[idx, AF]
    if len(X) < 200: continue
    med = X.median(); mad = (X - med).abs().median().replace(0, np.nan) * 1.4826
    rz.loc[idx] = ((X - med) / mad).clip(-50, 50).values
    xs = StandardScaler().fit_transform(X)
    iso = IsolationForest(n_estimators=150, contamination=CONTAM, random_state=SEED, n_jobs=1).fit(xs)
    score.loc[idx] = -iso.score_samples(xs)           # higher = more anomalous
    flag.loc[idx] = iso.predict(xs) == -1
# group flagged points into events (gap <= 15 min merged), keep events of >= 2 points (>= 10 min)
fi = flag[flag].index
events = []; cur = None
for t in fi:
    if cur and (t - cur[1]) <= pd.Timedelta("15min"): cur[1] = t; cur[2].append(t)
    else:
        if cur: events.append(cur)
        cur = [t, t, [t]]
if cur: events.append(cur)
sd_starts = shut["start"].values
ev = []
for a, b, pts in events:
    if len(pts) < 2: continue
    seg = slice(a, b); z = rz.loc[pts].abs().mean().sort_values(ascending=False)
    top = z.index[:3].tolist(); st = lab.loc[pts].map(names).mode().iloc[0]
    nxt = sd_starts[sd_starts > np.datetime64(b)]; mins_to = float((nxt[0] - np.datetime64(b)) / np.timedelta64(1, "m")) if len(nxt) else np.nan
    ev.append(dict(start=a, end=b, duration_min=(b - a) / STEP * 5 + 5, n_flagged_points=len(pts), state=st,
                   peak_score=round(float(score.loc[pts].max()), 3), mean_score=round(float(score.loc[pts].mean()), 3),
                   top_variables="; ".join(f"{t}({rz.loc[pts, t].mean():+.1f}σ)" for t in top),
                   minutes_to_next_shutdown=mins_to))
ev = pd.DataFrame(ev); ev["severity"] = ev["mean_score"] * np.log1p(ev["n_flagged_points"])
ev.sort_values("start").to_csv(os.path.join(HERE, "anomalous_periods.csv"), index=False)
INSIGHT.update(n_anomaly_events=len(ev), n_flagged_points=int(flag.sum()))
# anomaly rate by state and pre-shutdown enrichment
per_state = pd.DataFrame({"points": lab.map(names).value_counts(), "flagged": lab[flag.index[flag]].map(names).value_counts()}).fillna(0)
per_state["rate_%"] = (100 * per_state.flagged / per_state.points).round(2)
INSIGHT["anomaly_rate_by_state"] = per_state["rate_%"].to_dict()
ev_state = ev.groupby("state").size().to_dict(); INSIGHT["events_by_state"] = ev_state
# is an anomaly event more likely than chance to end within 1 h before a shutdown? (events inside outages excluded, see ctx below)
ev["context"] = "in operation"                                           # provisional; refined after ctx() is defined
act_pts = (~is_down) & (~data_gap); near = pd.Series(False, index=clean.index)
for a_ in shut.start: near.loc[a_ - pd.Timedelta("60min"):a_ - STEP] = True
baseline = float((near & act_pts).sum() / act_pts.sum())              # chance that a random operating timestamp is within 60 min before a shutdown
INSIGHT["events_within_30m_before_shutdown"] = int((ev["minutes_to_next_shutdown"] <= 30).sum())

# choose 5 diverse, high-severity events (distinct driver family) + 1 event that precedes a shutdown
FAM = {"Cyclone_Material_Temp": "material", "Tmat_minus_in": "material", "Cyclone_Inlet_Draft": "draft", "Cyclone_Outlet_Gas_draft": "draft",
       "Cyclone_cone_draft": "draft", "dDraft_15m": "draft", "Cyclone_Inlet_Gas_Temp": "gas_temp", "Cyclone_Gas_Outlet_Temp": "gas_temp",
       "T_in_minus_out": "gas_temp", "dT_in_15m": "gas_temp", "std_T_1h": "volatility"}
def ctx(r):
    b = state_full.loc[r.start - pd.Timedelta("30min"):r.start - STEP]; a_ = state_full.loc[r.end + STEP:r.end + pd.Timedelta("30min")]
    return "fan-on blip inside an outage" if (len(b) and len(a_) and b.isin(["SHUTDOWN", "DATA_GAP"]).mean() > .8 and a_.isin(["SHUTDOWN", "DATA_GAP"]).mean() > .8) else "in operation"
ev["context"] = ev.apply(ctx, axis=1)
inop = ev[ev.context == "in operation"]; n_in = len(inop); k_in = int((inop.minutes_to_next_shutdown <= 60).sum()); ph = k_in / n_in
z_ = 1.96; den = 1 + z_**2 / n_in; ctr = (ph + z_**2 / (2 * n_in)) / den; hw = z_ * np.sqrt(ph * (1 - ph) / n_in + z_**2 / (4 * n_in**2)) / den
INSIGHT.update(events_in_operation=n_in, events_within_1h_before_shutdown=k_in, events_within_1h_before_shutdown_pct=round(100 * ph, 1),
               events_within_1h_ci95_pct=[round(100 * (ctr - hw), 1), round(100 * (ctr + hw), 1)], baseline_pct_within_1h_before_shutdown=round(100 * baseline, 2),
               enrichment_x=round(ph / baseline, 1), enrichment_x_ci95=[round((ctr - hw) / baseline, 1), round((ctr + hw) / baseline, 1)])
ev.sort_values("start").to_csv(os.path.join(HERE, "anomalous_periods.csv"), index=False)
INSIGHT["events_inside_outages"] = int((ev.context != "in operation").sum())
def hypothesis(r):
    a, b = r.start, r.end; pts = slice(a, b)
    base = clean.loc[a - pd.Timedelta("90min"):a - STEP]; during = clean.loc[pts]
    d = {v: during[v].mean() - base[v].mean() for v in VARS}
    sd_ = {v: d[v] / (base[v].std() + 1e-6) for v in VARS}
    dr = np.mean([d["Cyclone_Inlet_Draft"], d["Cyclone_Outlet_Gas_draft"], d["Cyclone_cone_draft"]])
    cone_gap = (during["Cyclone_cone_draft"] - during["Cyclone_Outlet_Gas_draft"]).mean() - (base["Cyclone_cone_draft"] - base["Cyclone_Outlet_Gas_draft"]).mean()
    dTin, dTout, dTm = d[TEMP], d["Cyclone_Gas_Outlet_Temp"], d["Cyclone_Material_Temp"]
    ev_txt = (f"Δ vs prior 90 min: inlet T {dTin:+.0f}, outlet T {dTout:+.0f}, material T {dTm:+.0f}, "
              f"inlet draft {d['Cyclone_Inlet_Draft']:+.0f}, outlet draft {d['Cyclone_Outlet_Gas_draft']:+.0f}, cone draft {d['Cyclone_cone_draft']:+.0f}")
    mq = during["Cyclone_Material_Temp"].quantile([.1, .9]); flat_mat = (mq[.9] - mq[.1]) < 5 and during["Cyclone_Material_Temp"].median() < 150 and base["Cyclone_Material_Temp"].mean() > 500
    flat_any = [v for v in VARS if during[v].std() < 0.3 and len(during) >= 6]
    base_in = base[TEMP].mean()
    if flat_mat and abs(dTin) < 40:
        k = "material_flatline"; h = "Material temperature flat-lines near ambient (~40) for the whole event while gas temperatures and all drafts stay normal: thermocouple / transmitter fault (open circuit or lost signal), not a process upset."
    elif flat_any:
        k = "frozen_tag"; h = f"{flat_any[0].replace('Cyclone_', '')} is frozen at a constant value while other tags keep moving: stuck sensor / stale signal from the DCS."
    elif dTout < -80 and abs(dTin) < 40:
        k = "outlet_collapse"; h = f"Outlet gas temperature falls by about {-dTout:.0f} C while inlet gas temperature changes by only {dTin:+.0f} C: heat loss between inlet and outlet (cold false-air ingress or a duct / cyclone leak" + (f"; drafts strengthening by about {-dr:.0f} fits extra air entering" if dr < -8 else "") + ") or an outlet thermocouple fault."
    elif dTin > 150 and base_in < 600:
        k = "warmup_ramp"; h = f"Inlet gas rises {dTin:+.0f} C from a cool baseline ({base_in:.0f} C in the prior 90 min) while outlet and material temperatures lag: a warm-up / restart heat-up ramp, an expected transient rather than a fault."
    elif dTin > 150 and dTout - dTin < -80:
        k = "inlet_surge"; h = f"Inlet gas temperature surges ({dTin:+.0f} C vs the prior 90 min) while outlet gas rises only {dTout:+.0f} C: a hot-gas transient the outlet has not yet followed (burner / kiln gas surge) or a spiking inlet thermocouple; check the raw DCS trend before acting."
    elif dr > 40 and dTin < -30:
        k = "draft_collapse"; h = "Draft collapse (draft rising toward zero) with falling temperatures: ID-fan loss / damper closure or an upstream trip, i.e. a pre-shutdown or aborted-restart pattern."
    elif dr < -25 and dTin < -15:
        k = "draft_strong_cooling"; h = "Draft strengthening (more negative) while gas temperature falls: surge of tramp/cold air (false-air ingress or open inspection door) or extra damper opening cooling the gas stream."
    elif dr < -25 and dTin > 15:
        k = "draft_strong_heating"; h = "Draft strengthening together with hotter inlet gas: sudden increase in gas throughput (fan speed-up / high kiln or burner firing) pulling hot gas through the cyclone."
    elif cone_gap > 25:
        k = "cone_blockage"; h = "Cone draft diverging from outlet draft (cone becomes less negative than the outlet): partial material build-up or blockage in the cone / discharge, restricting flow."
    elif cone_gap < -25:
        k = "cone_leak"; h = "Cone draft pulled more negative than the outlet draft: cone/dip-leg air leak or material discharge surge."
    elif dTm < -300 and abs(dTin) < 25:
        k = "material_dropout"; h = f"Material temperature falls {-dTm:.0f} C during the event while gas temperatures and drafts hold: a fall that large and fast is not physical for bulk material, so this is most likely a thermocouple / transmitter dropout (reading near ambient) rather than a feed change."
    elif dTm < -40 and abs(dTin) < 25:
        k = "material_drop"; h = "Material temperature drops while gas temperatures stay flat: change in material feed/residence (cold-feed surge, material flow interruption) or a material thermocouple fault."
    elif dTin > 30:
        k = "inlet_heating"; h = "Inlet gas heating-up transient without matching draft change: upstream burner/process disturbance (higher firing rate) rather than a cyclone-side problem."
    elif dTin < -30:
        k = "inlet_falling"; h = "Inlet gas temperature falling without matching draft change: upstream heat-source drop (reduced firing / feed disturbance), cyclone hydraulics unchanged."
    elif during[TEMP].std() > 2.5 * base[TEMP].std():
        k = "oscillation"; h = f"Inlet temperature oscillates (std {during[TEMP].std():.0f} vs {base[TEMP].std():.0f} in the prior 90 min) around an unchanged mean ({dTin:+.0f} C): possible hunting of the burner / feed control loop or pulsing material feed, not a level shift. Low-confidence hypothesis; confirm on the raw trend."
    elif dr < -10 and dTin < -5 and dTm < -10:
        k = "all_sag"; h = "All temperatures sag slightly while all drafts strengthen together: higher air flow with unchanged heat input (fan speed-up, damper opening or false-air), i.e. a set-point/throughput change rather than a fault."
    else:
        k = "generic"; h = "Short-lived multivariate excursion with small mean shifts: process noise or sensor instability (check for erratic readings; high rolling variance)."
    pre_txt = f"; shutdown began {r.minutes_to_next_shutdown:.0f} min after the event ended" if r.minutes_to_next_shutdown <= 180 else ""
    return h, ev_txt + pre_txt, k

cand = ev[(ev.duration_min >= 15) & (ev.context == "in operation")].sort_values("severity", ascending=False).head(250)
pick = []; used = set()
for _, r in cand.iterrows():                      # one event per distinct physical signature, never the generic fallback
    h_, e_, k_ = hypothesis(r)
    if k_ in used or k_ == "generic": continue
    if any(abs((r.start - p.start).days) < 3 for p in pick): continue
    pick.append(r); used.add(k_)
    if len(pick) == 5: break
if not any(p.minutes_to_next_shutdown <= 45 for p in pick):
    pre_c = cand[(cand.minutes_to_next_shutdown <= 45) & (cand.n_flagged_points >= 3)]
    if len(pre_c): pick.append(pre_c.sort_values("severity", ascending=False).iloc[0])
pick = pd.DataFrame(pick).reset_index(drop=True)
hyp = []
for i, r in pick.iterrows():
    h, e, sig = hypothesis(r)
    pre_state = state_full.loc[r.start - pd.Timedelta("30min"):r.start - STEP].mode(); post_state = state_full.loc[r.end + STEP:r.end + pd.Timedelta("30min")].mode()
    hyp.append(dict(id=i + 1, start=r.start, end=r.end, duration_min=r.duration_min, state=r.state, top_variables=r.top_variables,
                    state_before=pre_state.iloc[0] if len(pre_state) else "", state_after=post_state.iloc[0] if len(post_state) else "",
                    evidence=e, signature=sig, root_cause_hypothesis=h))
    # plot with context
    a, b = r.start - pd.Timedelta("4h"), r.end + pd.Timedelta("4h"); d = clean.loc[a:b]; s = state_full.loc[a:b]
    fig, axs = plt.subplots(4, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1.5, .4]})
    for v in [TEMP, "Cyclone_Gas_Outlet_Temp", "Cyclone_Material_Temp"]: axs[0].plot(d.index, d[v], lw=.9, label=v.replace("Cyclone_", ""))
    for v in ["Cyclone_Inlet_Draft", "Cyclone_Outlet_Gas_draft", "Cyclone_cone_draft"]: axs[1].plot(d.index, d[v], lw=.9, label=v.replace("Cyclone_", ""))
    sc = score.reindex(d.index); axs[2].plot(d.index, sc, lw=.7, color="k"); axs[2].set_ylabel("IF anomaly score")
    fl_ = flag.reindex(d.index).fillna(False); axs[2].scatter(d.index[fl_], sc[fl_], color="red", s=8)
    axs[3].scatter(s.index, [0]*len(s), c=[pal.get(x, (0, 0, 0)) for x in s], marker="|", s=250); axs[3].set_yticks([])
    for ax in axs[:3]: ax.axvspan(r.start, r.end + STEP, color="orange", alpha=.25)
    axs[0].legend(ncol=3, loc="lower left", fontsize=8); axs[1].legend(ncol=3, loc="lower left", fontsize=8)
    import textwrap
    axs[0].set_title(f"Anomaly #{i+1}  {r.start}  ({r.duration_min:.0f} min, state: {r.state})\n{r.top_variables}\nHypothesis: " + "\n".join(textwrap.wrap(h, 140)), fontsize=9, loc="left")
    axs[3].set_xlabel("state strip: " + ", ".join(sorted(set(s))), fontsize=8); save(f"10_anomaly_{i+1}.png")
pd.DataFrame(hyp).to_csv(os.path.join(HERE, "root_cause_hypotheses.csv"), index=False)
hrs = lab.map(names).value_counts() * 5 / 60
ev_rate = (ev[ev.context == "in operation"].groupby("state").size() / hrs * 100).round(2); INSIGHT["anomaly_events_per_100h_by_state"] = ev_rate.to_dict()
INSIGHT["anomaly_mean_event_min_by_state"] = ev.groupby("state").duration_min.mean().round(1).to_dict()
fig, ax = plt.subplots(figsize=(10, 4)); ev_rate.sort_values().plot.barh(ax=ax, color="darkorange"); ax.set_xlabel("anomaly events per 100 operating hours"); ax.set_title("Contextual anomaly event rate by state"); save("11_anomaly_rate_by_state.png")

# ---- backtest of the proposed early-warning rule (trailing 15-min changes only, active operation only)
rule = (F["dDraft_15m"] > 40) & (F["dT_in_15m"] < -30) & (~is_down) & (~data_gap)
fire = rule[rule].index; alerts = []
for t in fire:
    if alerts and (t - alerts[-1]) <= pd.Timedelta("30min"): continue          # one alert per burst (30-min refractory)
    alerts.append(t)
alerts = pd.DatetimeIndex(alerts); op_hours = float(act_pts.sum() * 5 / 60)
ss = shut.start.values; leads = []; hit = 0
for a_ in shut.start:
    w_ = fire[(fire >= a_ - pd.Timedelta("60min")) & (fire < a_)]
    if len(w_): hit += 1; leads.append((a_ - w_[0]) / pd.Timedelta("1min"))
good = sum(((ss >= np.datetime64(t)) & (ss <= np.datetime64(t + pd.Timedelta("60min")))).any() for t in alerts)
INSIGHT["early_warning_rule"] = dict(rule="inlet-draft rise > 40 toward zero AND inlet T fall > 30 C, both over 15 min, machine active",
    shutdowns_total=len(shut), shutdowns_with_alert_in_prior_hour=hit, hit_rate_pct=round(100 * hit / len(shut), 1),
    median_lead_min=float(np.median(leads)) if leads else None, alerts_total=len(alerts), alerts_followed_by_shutdown_within_1h=int(good),
    alert_precision_pct=round(100 * good / max(len(alerts), 1), 1), false_alerts_per_100_operating_h=round(100 * (len(alerts) - good) / op_hours, 2),
    operating_hours=round(op_hours))

# ------------------------------------------------------------------ 6. FORECASTING (Inlet gas temp, 12 steps = 1 h)
log("forecasting")
H = 12
s = clean[TEMP]
X = pd.DataFrame(index=clean.index)
for l in range(12): X[f"lag{l}"] = s.shift(l)
for w_ in (6, 12, 36): X[f"rm{w_}"] = s.rolling(w_).mean()
X["rstd12"] = s.rolling(12).std(); X["d3"] = s.diff(3); X["d12"] = s.diff(12)
for v in VARS[1:]: X[v] = clean[v]; X[v + "_d3"] = clean[v].diff(3)
X["down_now"] = is_down.astype(int); X["hr_sin"] = np.sin(2 * np.pi * clean.index.hour / 24); X["hr_cos"] = np.cos(2 * np.pi * clean.index.hour / 24)
Y = pd.concat({h: s.shift(-h) for h in range(1, H + 1)}, axis=1)
ok = X.notna().all(axis=1) & Y.notna().all(axis=1)
Xo, Yo = X[ok], Y[ok]
cut = Xo.index[int(len(Xo) * 0.8)]
tr = Xo.index < cut - pd.Timedelta("1h"); te = Xo.index >= cut
Xtr, Xte, Ytr, Yte = Xo[tr], Xo[te], Yo[tr], Yo[te]
last_tr, last_te = Xtr["lag0"].values, Xte["lag0"].values
INSIGHT.update(forecast_train_rows=int(tr.sum()), forecast_test_rows=int(te.sum()), test_start=str(Xte.index[0]), test_end=str(Xte.index[-1]))
preds = {}
preds["persistence"] = np.repeat(last_te[:, None], H, 1)
preds["moving_avg_6"] = np.repeat(Xte["rm6"].values[:, None], H, 1)
sc_ = StandardScaler().fit(Xtr)
rd = Ridge(alpha=10).fit(sc_.transform(Xtr), Ytr.values - last_tr[:, None])
preds["ridge_AR"] = rd.predict(sc_.transform(Xte)) + last_te[:, None]
hgb = np.zeros((len(Xte), H)); tr_sub = np.arange(0, len(Xtr), 2); hgb_models = []
for h in range(H):
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, random_state=SEED)
    m.fit(Xtr.values[tr_sub], (Ytr.values[:, h] - last_tr)[tr_sub]); hgb_models.append(m); hgb[:, h] = m.predict(Xte.values) + last_te
    log(f"  hgb horizon {h+1}/{H}")
preds["hist_gradient_boosting"] = hgb
yt = Yte.values
# regimes by what the +-1 h window around the forecast origin contains: all-active / mixed (transition) / all-down
dn = is_down.astype(float)
frac_down = dn.rolling(25, center=True, min_periods=1).mean().reindex(Xte.index).values
reg = {"all": np.ones(len(Xte), bool), "steady_operation": frac_down == 0, "shutdown_transition": (frac_down > 0) & (frac_down < 1), "inside_shutdown": frac_down == 1}
mrows = []
for mname, p in preds.items():
    for rg, mk in reg.items():
        e = (p[mk] - yt[mk]); mrows.append(dict(model=mname, regime=rg, horizon="1-12 avg", n=int(mk.sum()), RMSE=np.sqrt((e**2).mean()), MAE=np.abs(e).mean()))
        if rg == "all":
            for h in (1, 3, 6, 12):
                eh = p[:, h-1] - yt[:, h-1]; mrows.append(dict(model=mname, regime=rg, horizon=f"h={h} ({5*h} min)", n=len(eh), RMSE=np.sqrt((eh**2).mean()), MAE=np.abs(eh).mean()))
metrics = pd.DataFrame(mrows).round(3); metrics.to_csv(os.path.join(HERE, "forecast_metrics.csv"), index=False)
INSIGHT["forecast_metrics"] = metrics[(metrics.horizon == "1-12 avg")].pivot(index="model", columns="regime", values="RMSE").round(2).to_dict()
def _m(model, hz, col): return float(metrics[(metrics.model == model) & (metrics.regime == "all") & (metrics.horizon == hz)][col].iloc[0])
INSIGHT["forecast_60min"] = {m_: dict(RMSE=_m(m_, "h=12 (60 min)", "RMSE"), MAE=_m(m_, "h=12 (60 min)", "MAE")) for m_ in preds}
INSIGHT["forecast_5min"] = {m_: dict(RMSE=_m(m_, "h=1 (5 min)", "RMSE"), MAE=_m(m_, "h=1 (5 min)", "MAE")) for m_ in preds}
INSIGHT["ma6_share_of_hgb_gain_rmse_pct"] = {"avg_5_60min": round(100 * (_m("persistence", "1-12 avg", "RMSE") - _m("moving_avg_6", "1-12 avg", "RMSE")) / (_m("persistence", "1-12 avg", "RMSE") - _m("hist_gradient_boosting", "1-12 avg", "RMSE")), 0),
                                              "at_60min": round(100 * (_m("persistence", "h=12 (60 min)", "RMSE") - _m("moving_avg_6", "h=12 (60 min)", "RMSE")) / (_m("persistence", "h=12 (60 min)", "RMSE") - _m("hist_gradient_boosting", "h=12 (60 min)", "RMSE")), 0)}
INSIGHT["forecast_mae_all"] = metrics[(metrics.horizon == "1-12 avg") & (metrics.regime == "all")].set_index("model")["MAE"].to_dict()
INSIGHT["forecast_rmse_all"] = metrics[(metrics.horizon == "1-12 avg") & (metrics.regime == "all")].set_index("model")["RMSE"].to_dict()
# non-stationarity: how the operating level and the forecast error move from year to year
yrs = Xte.index.year; drift = []
act_t = clean.loc[act_pts, TEMP]
for y_ in sorted(set(clean.index.year)):
    a_t = act_t[act_t.index.year == y_]; row = dict(year=int(y_), operating_hours=round(len(a_t) * 5 / 60), inlet_T_mean=round(a_t.mean(), 1), inlet_T_std=round(a_t.std(), 1),
        inlet_T_p05=round(a_t.quantile(.05), 1), inlet_T_p95=round(a_t.quantile(.95), 1), downtime_h=round(float(shut[shut.start.dt.year == y_].duration_hours.sum())),
        draft_in_mean=round(clean.loc[act_pts & (clean.index.year == y_), DRAFT].mean(), 1))
    mk_ = (yrs == y_)
    if mk_.sum() > 0:
        row["test_origins"] = int(mk_.sum())
        for m_, p in preds.items(): row[f"rmse_{m_}"] = round(float(np.sqrt(((p[mk_] - yt[mk_]) ** 2).mean())), 2)
    drift.append(row)
drift = pd.DataFrame(drift); drift.to_csv(os.path.join(HERE, "yearly_drift.csv"), index=False); INSIGHT["yearly_drift"] = drift.to_dict(orient="records")
# forecasts.csv: long format, every 12th origin (hourly) to keep file size manageable; metrics use ALL origins
sel = np.arange(0, len(Xte), 12); out = []
for h in range(1, H + 1):
    d_ = pd.DataFrame({"origin_time": Xte.index[sel], "horizon_step": h, "target_time": Xte.index[sel] + h * STEP, "y_true": yt[sel, h-1]})
    for mname, p in preds.items(): d_[mname] = p[sel, h-1].round(2)
    out.append(d_)
pd.concat(out).sort_values(["origin_time", "horizon_step"]).to_csv(os.path.join(HERE, "forecasts.csv"), index=False)
# plots
fig, ax = plt.subplots(figsize=(8, 4.5))
for mname, p in preds.items(): ax.plot(range(1, H + 1), np.sqrt(((p - yt) ** 2).mean(0)), "o-", label=mname)
ax.set_xlabel("horizon (steps of 5 min)"); ax.set_ylabel("RMSE"); ax.set_title("RMSE by forecast horizon (test set)"); ax.legend(); save("12_forecast_rmse_by_horizon.png")
# example: best steady window and worst (transition) window for h=12
fig, axs = plt.subplots(2, 1, figsize=(13, 7))
for ax, (title, mk) in zip(axs, [("Steady operation", reg["steady_operation"]), ("Shutdown transition", reg["shutdown_transition"])]):
    idxs = np.where(mk)[0]; pos = idxs[len(idxs) // 2] if title.startswith("Steady") else idxs[np.argmax(np.abs(preds["persistence"][idxs, 11] - yt[idxs, 11]))]
    lo, hi = max(pos - 72, 0), min(pos + 72, len(Xte) - 1); tt = Xte.index[lo:hi]
    ax.plot(tt + 12 * STEP, yt[lo:hi, 11], "k", lw=1.5, label="actual (t+60min)")
    for mname in ("persistence", "ridge_AR", "hist_gradient_boosting"): ax.plot(tt + 12 * STEP, preds[mname][lo:hi, 11], lw=1, label=mname)
    ax.set_title(f"{title}: 60-min-ahead forecast vs actual"); ax.legend(ncol=4, fontsize=8)
save("13_forecast_examples.png")

# forecast for the hour after the last complete observation in the file (gradient-boosting vs persistence)
Xl = X.dropna().iloc[[-1]]; t_l = Xl.index[0]; l0 = float(Xl["lag0"].iloc[0])
nxt = pd.DataFrame({"origin_time": t_l, "horizon_step": range(1, H + 1), "target_time": [t_l + k * STEP for k in range(1, H + 1)],
                    "persistence": l0, "hist_gradient_boosting": [round(float(m.predict(Xl.values)[0]) + l0, 2) for m in hgb_models],
                    "machine_down_at_origin": bool(is_down.loc[t_l])})
nxt.to_csv(os.path.join(HERE, "next_1_hour_forecast.csv"), index=False)
fig, ax = plt.subplots(figsize=(10, 4)); hist_ = s.loc[t_l - pd.Timedelta("6h"):t_l]
ax.plot(hist_.index, hist_, "k", lw=1.2, label="recent inlet gas T"); ax.plot(nxt.target_time, nxt.hist_gradient_boosting, "o-", label="forecast (HGB)")
ax.plot(nxt.target_time, nxt.persistence, "--", label="persistence"); ax.legend(); ax.set_title(f"Next-hour forecast from {t_l}")
save("14_next_hour_forecast.png")

# ------------------------------------------------------------------ 6b. EXTRAS (added after review): interval forecasts, state stability over time, second anomaly detector
log("extras: forecast intervals")
# (a) 10th-90th percentile bands for the HGB forecast: one quantile-loss model per quantile, on horizons 5, 15, 30 and 60 min (nominal coverage 80 %)
QH = (1, 3, 6, 12); qrows = []; qband = {}
for h in QH:
    qm = {}
    for qq in (0.1, 0.9):
        m = HistGradientBoostingRegressor(loss="quantile", quantile=qq, max_iter=200, learning_rate=0.08, max_leaf_nodes=31, random_state=SEED)
        m.fit(Xtr.values[tr_sub], (Ytr.values[:, h - 1] - last_tr)[tr_sub]); qm[qq] = m.predict(Xte.values) + last_te
    lo_, hi_ = np.minimum(qm[0.1], qm[0.9]), np.maximum(qm[0.1], qm[0.9]); qband[h] = (lo_, hi_)
    for rg, mk in reg.items():
        inside = (yt[mk, h - 1] >= lo_[mk]) & (yt[mk, h - 1] <= hi_[mk])
        qrows.append(dict(horizon=f"h={h} ({5*h} min)", regime=rg, n=int(mk.sum()), nominal_coverage=0.8, coverage=round(float(inside.mean()), 3),
                          mean_width_C=round(float((hi_[mk] - lo_[mk]).mean()), 1), below_10pct_share=round(float((yt[mk, h - 1] < lo_[mk]).mean()), 3),
                          above_90pct_share=round(float((yt[mk, h - 1] > hi_[mk]).mean()), 3)))
qdf = pd.DataFrame(qrows); qdf.to_csv(os.path.join(HERE, "forecast_intervals.csv"), index=False)
INSIGHT["forecast_interval_coverage_60min"] = {r_["regime"]: dict(coverage=r_["coverage"], width=r_["mean_width_C"]) for r_ in qrows if r_["horizon"].startswith("h=12")}
fig, ax = plt.subplots(figsize=(13, 4)); idxs = np.where(reg["steady_operation"])[0]; pos = idxs[len(idxs) // 2]; lo, hi = max(pos - 72, 0), min(pos + 72, len(Xte) - 1); tt = Xte.index[lo:hi] + 12 * STEP
ax.fill_between(tt, qband[12][0][lo:hi], qband[12][1][lo:hi], color="tab:blue", alpha=.25, label="10th-90th percentile band")
ax.plot(tt, yt[lo:hi, 11], "k", lw=1.3, label="actual (t+60 min)"); ax.plot(tt, hgb[lo:hi, 11], color="tab:blue", lw=1, label="HGB median-type forecast"); ax.legend(ncol=3, fontsize=8)
ax.set_title("60-min-ahead forecast with 10th-90th percentile band (steady operation, test set)"); save("15_forecast_interval_band.png")

log("extras: state stability over time")
# (b) state shares per year, per-year refits and per-year state centres
yr = pd.Series(Ga.index.year, index=Ga.index); lab_n = lab.map(names)
share = pd.crosstab(yr, lab_n, normalize="index").mul(100).round(1)
rows_b = []
for y_ in sorted(yr.unique()):
    mk = (yr == y_).values
    if mk.sum() < 3000: continue
    my = KMeans(K, n_init=5, random_state=SEED).fit(Z[mk]); ari_y = adjusted_rand_score(km.predict(Z[mk]), my.labels_)
    row = dict(year=int(y_), active_points=int(mk.sum()), ari_year_refit_vs_global_model=round(float(ari_y), 3))
    row.update({f"share_%_{n_}": float(share.loc[y_, n_]) if n_ in share.columns else 0.0 for n_ in names.values()})
    for n_ in names.values():
        c_ = [k_ for k_, v_ in names.items() if v_ == n_][0]; sel_ = mk & (lab.values == c_)
        row[f"inletT_mean_{n_}"] = round(float(Ga.loc[sel_, "T_in"].mean()), 1) if sel_.sum() else np.nan
    rows_b.append(row)
sb = pd.DataFrame(rows_b); sb.to_csv(os.path.join(HERE, "state_stability_by_year.csv"), index=False)
INSIGHT["state_share_by_year_pct"] = share.to_dict(orient="index"); INSIGHT["ari_year_refit_vs_global"] = {int(r_["year"]): r_["ari_year_refit_vs_global_model"] for r_ in rows_b}
fig, ax = plt.subplots(figsize=(10, 4)); share.plot.bar(stacked=True, ax=ax, colormap="tab10"); ax.set_ylabel("% of active time"); ax.set_xlabel("year"); ax.set_title("State shares per calendar year (global model)"); ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, .5)); save("16_state_share_by_year.png")

log("extras: second anomaly detector (robust Mahalanobis)")
# (c) cross-check: state-conditional robust Mahalanobis distance (MinCovDet, 9 features: raw tags + 15-min deltas + 1-h volatility; the two
# difference features used by the Isolation Forest are left out because they are linear combinations of the raw tags and make the covariance singular)
from sklearn.covariance import MinCovDet
from scipy.stats import chi2
MF = VARS + ["dT_in_15m", "dDraft_15m", "std_T_1h"]; md_ = pd.Series(np.nan, index=lab.index); cut_ = chi2.ppf(0.99, len(MF))
for c, nm in names.items():
    idx = lab.index[lab == c]
    if len(idx) < 500: continue
    Xs = F.loc[idx, MF]; mu, sd_ = Xs.median(), Xs.std().replace(0, 1); Xn = ((Xs - mu) / sd_).values
    sub_ = np.random.RandomState(SEED).choice(len(Xn), min(6000, len(Xn)), replace=False)
    mcd = MinCovDet(support_fraction=0.75, random_state=SEED).fit(Xn[sub_]); md_.loc[idx] = mcd.mahalanobis(Xn)       # squared distance
def md_share(a, b):
    p = md_.loc[a:b].dropna(); return (np.nan, np.nan, 0) if len(p) == 0 else (float((p > cut_).mean()), float(np.sqrt(p.median())), len(p))
xr = []
for _, r in ev.iterrows():
    sh_, med_, n_ = md_share(r.start, r.end); xr.append(dict(start=r.start, share_points_beyond_chi2_99=sh_, median_robust_distance=med_, n_points=n_))
xr = pd.DataFrame(xr); evx = pd.concat([ev.reset_index(drop=True)[["start", "end", "duration_min", "state", "context", "severity"]], xr.drop(columns="start")], axis=1)
evx["mahalanobis_agrees"] = evx["share_points_beyond_chi2_99"] >= 0.5
evx.to_csv(os.path.join(HERE, "anomaly_crosscheck_mahalanobis.csv"), index=False)
evalu = evx[(evx.n_points > 0) & (evx.context == "in operation")]
base_rate = float((md_.dropna() > cut_).mean())
hx = []
for h_ in hyp:
    sh_, med_, n_ = md_share(h_["start"], h_["end"]); hx.append(dict(id=h_["id"], signature=h_["signature"], duration_min=h_["duration_min"], share_points_beyond_chi2_99=round(sh_, 2), median_robust_distance=round(med_, 1), agrees=bool(sh_ >= 0.5)))
hx = pd.DataFrame(hx); hx.to_csv(os.path.join(HERE, "anomaly_crosscheck_selected5.csv"), index=False)
# control: random windows of the same length in the same state (an event-like window picked at random), same >= 50 % rule
rs_ = np.random.RandomState(SEED); null_hits = []; mdv = md_.dropna()
for _, r in evalu.iterrows():
    npts = max(int(r["duration_min"] // 5), 2); cand = mdv.index[(lab.map(names).reindex(mdv.index) == r["state"]).values]
    for _i in range(5):
        a0 = cand[rs_.randint(0, len(cand))]; w_ = mdv.loc[a0:a0 + (npts - 1) * STEP]; null_hits.append(float((w_ > cut_).mean() >= 0.5) if len(w_) else np.nan)
null_rate = float(np.nanmean(null_hits))
INSIGHT["mahalanobis_crosscheck"] = dict(random_window_agreement_pct=round(100 * null_rate, 1), cutoff_chi2_99=round(float(cut_), 1), baseline_share_points_beyond_cutoff=round(100 * base_rate, 2), events_evaluated=int(len(evalu)),
    events_agreeing_pct=round(100 * float(evalu["mahalanobis_agrees"].mean()), 1), selected5=hx.to_dict(orient="records"))

# ------------------------------------------------------------------ 7. INSIGHT DATA + WRITE-UP
# shutdown periodicity & what happens in the hour before
INSIGHT["shutdowns_per_year"] = shut.groupby(shut.start.dt.year).size().to_dict()
INSIGHT["downtime_h_per_year"] = shut.groupby(shut.start.dt.year).duration_hours.sum().round(1).to_dict()
INSIGHT["cluster_state_pct"] = cs.set_index("state")["pct_of_active_time"].to_dict()
INSIGHT["cluster_episode_median_min"] = cs.set_index("state")["episode_median_min"].to_dict()
# state in the 30 min before each shutdown vs overall
before = []
for a in shut.start:
    w_ = state_full.loc[a - pd.Timedelta("30min"):a - STEP]; before += w_[~w_.isin(["SHUTDOWN", "DATA_GAP"])].tolist()
INSIGHT["state_before_shutdown_pct"] = (pd.Series(before).value_counts(normalize=True) * 100).round(1).to_dict()
# state right after restart
after = []
for b in shut.end_exclusive:
    w_ = state_full.loc[b:b + pd.Timedelta("60min")]; after += w_[~w_.isin(["SHUTDOWN", "DATA_GAP"])].tolist()
INSIGHT["state_first_hour_after_restart_pct"] = (pd.Series(after).value_counts(normalize=True) * 100).round(1).to_dict()
INSIGHT["runtime_s"] = round(time.time() - t0)
json.dump(INSIGHT, open(os.path.join(HERE, "insights_numbers.json"), "w"), indent=1, default=str)
ew = INSIGHT["early_warning_rule"]; I = INSIGHT; f60 = I["forecast_60min"]; sh = I["shutdown_type_counts"]; drift_ = pd.DataFrame(I["yearly_drift"]).set_index("year")
worst = max(I["downtime_h_per_year"], key=I["downtime_h_per_year"].get); un = [k_ for k_ in I["cluster_state_pct"] if k_.startswith("Unstable")][0]
dg = [k_ for k_ in I["cluster_state_pct"] if k_.startswith("Degraded")][0]; pre_ = I["state_before_shutdown_pct"]; stab = I["k_stability"]
faults = [h_ for h_ in hyp if h_["signature"] in ("material_flatline", "material_dropout")]
fault_txt = (f"Anomalies #{', #'.join(str(h_['id']) for h_ in faults)} ({'; '.join(h_['signature'].replace('_', ' ') + f' for {h_['duration_min']:.0f} min' for h_ in faults)}) show the material temperature leaving the process while gas temperatures and drafts stay normal: instrument faults, not process events. " if faults else "")
ew_word = "usable as a monitoring signal but not as a trip predictor" if ew["hit_rate_pct"] < 60 or ew["alert_precision_pct"] < 50 else "a reasonable early warning"
md = f"""# Task 1 - Insights & recommendations

Generated by `task1_analysis.py` from `insights_numbers.json`; do not edit by hand (edits are overwritten on the next run). Forecast test period: {I['test_start'][:10]} to {I['test_end'][:10]}.

1. **The line is down about a fifth of the time.** {len(shut)} shutdown/idle events, {tot_h:,.0f} h ({tot_h/24:.0f} days, {100*is_down.mean():.1f} % of the record). {sh['short_stop(<2h)']} are short stops under 2 h; {sh['extended_outage(>48h)']} are outages over 48 h (longest {I['longest_shutdown_days']} days) and drive most of the hours. {worst} was the worst year ({I['downtime_h_per_year'][worst]:,.0f} h). **What counts as a shutdown:** the rule is median of the three drafts above -60 for 30+ min. That value sits in an almost empty valley (only {I['samples_between_-100_and_-40_pct']} % of samples lie between -100 and -40, plots/05b). {I['shutdowns_inlet_above_700C_most_of_length']} of the {len(shut)} events keep inlet gas above 700 C for most of their length, so many are fan-off / idle stops with hot gas rather than cold shutdowns; the drafts collapsing to about 0 is the reason they count.
2. **Restarts are a distinct, unstable regime that is also over-represented before shutdowns.** The unstable state is {I['cluster_state_pct'][un]:.1f} % of active time but {I['state_first_hour_after_restart_pct'].get(un, 0):.0f} % of the first hour after a restart (partly by construction, since it is defined by high volatility) and {pre_.get(un, 0):.1f} % of the 30 min before a shutdown (about {pre_.get(un, 0)/I['cluster_state_pct'][un]:.0f}x its normal share). Labels use trailing windows only. It is a moderate signal: about four in five pre-shutdown windows are in other states.
3. **Anomaly events are more common just before shutdowns than chance.** {I['events_within_1h_before_shutdown_pct']} % ({I['events_within_1h_before_shutdown']} of {I['events_in_operation']} in-operation events; 95 % interval {I['events_within_1h_ci95_pct'][0]}-{I['events_within_1h_ci95_pct'][1]} %) end within 1 h before a shutdown, against {I['baseline_pct_within_1h_before_shutdown']} % for a random operating timestamp: about {I['enrichment_x']}x (interval {I['enrichment_x_ci95'][0]}-{I['enrichment_x_ci95'][1]}x). Events are autocorrelated, so the interval is optimistic.
4. **Some anomalies are instrument faults.** {fault_txt}The "{dg}" state ({I['cluster_state_pct'][dg]:.1f} % of active time) has outlet gas about 160 C below inlet and is worth checking for false air or an outlet thermocouple problem.
5. **Forecasting helps in steady operation, not at shutdown onset.** The 60-minute-ahead RMSE / MAE is {f60['hist_gradient_boosting']['RMSE']:.1f} / {f60['hist_gradient_boosting']['MAE']:.1f} for gradient boosting against {f60['persistence']['RMSE']:.1f} / {f60['persistence']['MAE']:.1f} for persistence. Averaged over all horizons from 5 to 60 min the figures are {I['forecast_rmse_all']['hist_gradient_boosting']:.1f} / {I['forecast_mae_all']['hist_gradient_boosting']:.1f} against {I['forecast_rmse_all']['persistence']:.1f} / {I['forecast_mae_all']['persistence']:.1f}. A plain 6-step moving average already captures about {I['ma6_share_of_hgb_gain_rmse_pct']['avg_5_60min']:.0f} % of the boosting gain (about {I['ma6_share_of_hgb_gain_rmse_pct']['at_60min']:.0f} % at 60 min), so most of the gain is smoothing sensor jitter. In windows that contain a shutdown start every model has RMSE {min(I['forecast_metrics']['shutdown_transition'].values()):.0f}-{max(I['forecast_metrics']['shutdown_transition'].values()):.0f}.
6. **The process drifts, so models need monitoring.** Mean active inlet temperature is {drift_.inlet_T_mean[2017]:.0f}, {drift_.inlet_T_mean[2018]:.0f}, {drift_.inlet_T_mean[2019]:.0f} and {drift_.inlet_T_mean[2020]:.0f} C in 2017-2020, but its spread changes a lot (std {drift_.inlet_T_std[2017]:.0f}, {drift_.inlet_T_std[2018]:.0f}, {drift_.inlet_T_std[2019]:.0f}, {drift_.inlet_T_std[2020]:.0f}); inlet draft is stronger in 2020 ({drift_.draft_in_mean[2020]:.0f} vs {drift_.draft_in_mean[2018]:.0f} in 2018) and 2019 has the most downtime. On the test years the forecast error is similar (boosting RMSE {drift_.rmse_hist_gradient_boosting[2019]:.1f} for the last six weeks of 2019, {drift_.rmse_hist_gradient_boosting[2020]:.1f} for 2020), so I see no degradation on this single split. The choice of k = 5 holds across random seeds (adjusted Rand index {stab['ari_other_seeds_mean']:.2f}) but not across time: models fitted on the first and second half agree only at ARI {stab['ari_first_vs_second_half_models_on_second_half']:.2f}. That suggests refitting the states on a rolling window, monitoring per state, and retraining the forecaster on a schedule.
7. **Checks added after review.** (a) The 10th-90th percentile band from quantile gradient boosting covers {I['forecast_interval_coverage_60min']['all']['coverage']*100:.0f} % of 60-minute-ahead test values overall against the 80 % intended ({I['forecast_interval_coverage_60min']['steady_operation']['coverage']*100:.0f} % in steady operation, so the band is wider than needed there, {I['forecast_interval_coverage_60min']['steady_operation']['width']:.0f} C on average) but only {I['forecast_interval_coverage_60min']['shutdown_transition']['coverage']*100:.0f} % in windows that contain a shutdown start (`forecast_intervals.csv`). The band does not warn about shutdowns. (b) State shares move a lot between years under the global model: the Normal state is {I['state_share_by_year_pct'][2017]['Normal']:.0f} %, {I['state_share_by_year_pct'][2018]['Normal']:.0f} %, {I['state_share_by_year_pct'][2019]['Normal']:.0f} % and {I['state_share_by_year_pct'][2020]['Normal']:.0f} % of active time in 2017-2020, and a model refitted on each year alone agrees with the global one only at ARI {min(I['ari_year_refit_vs_global'].values()):.2f}-{max(I['ari_year_refit_vs_global'].values()):.2f} (`state_stability_by_year.csv`). I did not test why; a level shift in inlet temperature is one candidate, since the states are partly defined by level. Compare state shares across years with care and refit on a rolling window (not done here). (c) A second detector, a state-conditional robust Mahalanobis distance (MinCovDet, chi-square 99 % cutoff), has at least half of the points beyond the cutoff in {I['mahalanobis_crosscheck']['events_agreeing_pct']:.0f} % of the {I['mahalanobis_crosscheck']['events_evaluated']} in-operation events and in all 5 selected events, against {I['mahalanobis_crosscheck']['random_window_agreement_pct']:.0f} % for random windows of the same length and state. The events are unusual under a second, differently built method, but both detectors see the same features, so this is a consistency check and not independent confirmation (`anomaly_crosscheck_*.csv`).

## Recommendations
- **Early-warning rule ({ew_word}).** Rule: inlet-draft rise above 40 toward zero together with an inlet-temperature fall above 30 C, both over 15 min, machine running. In a backtest on trailing windows it fires in the hour before {ew['shutdowns_with_alert_in_prior_hour']} of {ew['shutdowns_total']} shutdowns ({ew['hit_rate_pct']} %), with a median lead time of {ew['median_lead_min']:.0f} min, and {ew['alert_precision_pct']} % of its {ew['alerts_total']} alerts are followed by a shutdown within an hour ({ew['false_alerts_per_100_operating_h']} false alerts per 100 operating hours). Use it as a dashboard flag combined with state, not as an automatic trip predictor.
- **Treat the restart hour separately**: suppress ordinary anomaly alerts in the first hour after restart and use ramp-rate checks instead.
- **Data-quality alerts**: flag more than 3 consecutive "Not Connect / I/O Timeout" samples and tags frozen or pinned at limits (-185, 1375, about 40 for material T).
- **Forecasting**: run forecasts only while the machine is active, add a shutdown-risk classifier, and add fan current / speed and feed rate, which could explain trips the temperatures cannot predict.
- **Validate hypotheses** against maintenance logs and operator notes before acting on any root cause.
"""
open(os.path.join(HERE, "insights.md"), "w", encoding="utf-8").write(md)
log("done; wrote CSVs, plots, insights.md, insights_numbers.json")
print(json.dumps(INSIGHT, indent=1, default=str))
