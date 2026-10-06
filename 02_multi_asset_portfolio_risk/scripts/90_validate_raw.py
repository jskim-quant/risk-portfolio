"""Raw data validation -> data/metadata/validation_report.md"""
import re, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; RAW = ROOT/"data/raw"
OUT = ROOT/"data/metadata/validation_report.md"
L = []; flags = []
def h(t): L.append(f"\n## {t}\n")
def p(t=""): L.append(t)
def flag(level, msg): flags.append((level, msg)); p(f"- **{level}** {msg}")
def rd(rel): return pd.read_csv(RAW/rel, index_col=0, parse_dates=True)
def short(c): return re.sub(r"^YC\.B\.U2\.EUR\.4F\.|\.SV_C_YM", "", c)
def base(df, name):
    d = df.index.duplicated().sum(); mono = df.index.is_monotonic_increasing
    if d: flag("FAIL", f"{name}: {d} duplicate dates")
    if not mono: flag("FAIL", f"{name}: index not sorted")
def gaps(s, name, bdays=10):
    s = s.dropna(); g = s.index.to_series().diff().dt.days
    big = g[g > bdays*1.5]
    for dt, n in big.items():
        flag("INFO", f"{name}: gap {n}d ending {dt.date()}")
def jumps(s, name, thr, unit=""):
    d = s.dropna().diff(); big = d[d.abs() > thr]
    for dt, v in big.head(8).items(): flag("WARN", f"{name}: jump {v:+.3f}{unit} on {dt.date()}")
    if len(big) > 8: flag("WARN", f"{name}: … {len(big)} jumps total > {thr}{unit}")
def stale(s, name, n=5):
    x = s.dropna(); run = (x.diff() == 0).astype(int)
    grp = (run == 0).cumsum(); lens = run.groupby(grp).sum()
    if lens.max() >= n: flag("WARN", f"{name}: {lens.max()} consecutive unchanged values (ends {x.index[(grp==lens.idxmax())][-1].date()})")

p("# Raw Data Validation Report"); p(f"Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}")

# ---------- ETFs ----------
h("1. ETFs (Yahoo)")
p("| ETF | start | end | n | max \\|daily ret\\| (date) | AdjClose≤Close | div-adj gap p.a. |"); p("|---|---|---|---|---|---|---|")
etf_ret = {}
for f in sorted((RAW/"market/etf").glob("*.csv")):
    t = f.stem; df = rd(f"market/etf/{t}.csv"); base(df, t)
    a, c = df[f"{t}_AdjClose"], df[f"{t}_Close"]
    if (a <= 0).any() or (c <= 0).any(): flag("FAIL", f"{t}: non-positive price")
    r = a.pct_change().dropna(); etf_ret[t] = r
    m = r.abs().idxmax()
    ok = (a <= c*1.0001).mean()
    yrs = (a.index[-1]-a.index[0]).days/365.25
    gap = ((a.iloc[-1]/a.iloc[0])/(c.iloc[-1]/c.iloc[0]))**(1/yrs)-1
    p(f"| {t} | {a.index[0].date()} | {a.index[-1].date()} | {len(a)} | {r.abs().max():.1%} ({m.date()}) | {ok:.1%} | {gap:.2%} |")
    for dt, v in r[r.abs() > 0.15].items(): flag("WARN", f"{t}: daily return {v:+.1%} on {dt.date()}")
    stale(a, t, 4)
    bd = pd.bdate_range(a.index[0], a.index[-1]); miss = len(bd.difference(a.index))/len(bd)
    if miss > 0.05: flag("WARN", f"{t}: {miss:.1%} business days missing")
p("\n*div-adj gap* = AdjClose CAGR − Close CAGR ≈ dividend yield. Equity ~1.5–3%, bonds ~2–5%, GLD ≈ 0.")

# ---------- UST ----------
h("2. UST curve (FRED, par CMT)")
u = rd("rates/ust/ust_curve.csv"); base(u, "UST")
p("| tenor | start | end | n | min | max | max \\|Δ1d\\| bp |"); p("|---|---|---|---|---|---|---|")
for c in u:
    s = u[c].dropna(); p(f"| {c} | {s.index[0].date()} | {s.index[-1].date()} | {len(s)} | {s.min():.2f} | {s.max():.2f} | {s.diff().abs().max()*100:.0f} |")
    jumps(s, c, 0.75, "pp"); gaps(s, c)
inv = (u["DGS10"] - u["DGS2"]).dropna(); p(f"\n10Y−2Y inverted on {(inv<0).mean():.1%} of days (latest {inv.iloc[-1]*100:+.0f}bp)")

# ---------- ECB ----------
h("3. ECB curves")
ecb = {k: rd(f"rates/ecb/{k}.csv").rename(columns=short) for k in ["G_N_A_SR","G_N_A_PY","G_N_C_SR","G_N_C_PY"]}
for k, df in ecb.items():
    base(df, k); na = df.isna().mean().max()
    p(f"- {k}: {df.index[0].date()} → {df.index[-1].date()}, n={len(df)}, cols={df.shape[1]}, max NaN {na:.2%}, range [{df.min().min():.2f}, {df.max().max():.2f}]")
    for c in df: jumps(df[c], f"{k}.{c}", 0.5, "pp")
a10 = ecb["G_N_A_PY"].filter(like="PY_10Y").iloc[:,0]; c10 = ecb["G_N_C_PY"].filter(like="PY_10Y").iloc[:,0]
sp = (c10 - a10).dropna()
p(f"\nAll-gov − AAA 10Y par spread: mean {sp.mean()*100:.0f}bp, max {sp.max()*100:.0f}bp ({sp.idxmax().date()}), <0 on {(sp<0).mean():.1%} days")
if (sp < -0.05).mean() > 0.01: flag("WARN", "All-gov curve below AAA on >1% of days")
zp = (ecb["G_N_A_SR"].filter(like="SR_10Y").iloc[:,0] - a10).dropna()
p(f"AAA 10Y zero − par: mean {zp.mean()*100:.1f}bp, |max| {zp.abs().max()*100:.0f}bp (should be small, sign follows curve slope)")

# ---------- JGB ----------
h("4. JGB curve (MoF)")
j = rd("rates/jgb/jgb_curve.csv"); base(j, "JGB")
p("| tenor | start | n | min | max |"); p("|---|---|---|---|---|")
for c in j:
    s = j[c].dropna(); p(f"| {c} | {s.index[0].date()} | {len(s)} | {s.min():.3f} | {s.max():.3f} |"); jumps(s, c, 0.5, "pp")
neg = (j["JGB_10Y"] < 0); p(f"\n10Y negative on {neg.sum()} days ({neg[neg].index.min().date()} – {neg[neg].index.max().date()})")
gaps(j["JGB_10Y"], "JGB_10Y")

# ---------- Short rates ----------
h("5. Short rates")
sr = {"DFF": rd("short_rates/DFF.csv").iloc[:,0], "DTB3": rd("short_rates/DTB3.csv").iloc[:,0],
      "EONIA": rd("short_rates/EONIA.csv").iloc[:,0], "ESTR": rd("short_rates/ESTR.csv").iloc[:,0],
      "JPY_CALL": rd("short_rates/JPY_CALL_ON.csv").iloc[:,0], "JPY_OECD_M": rd("short_rates/IRSTCI01JPM156N.csv").iloc[:,0]}
for k, s in sr.items():
    s = s.dropna(); p(f"- {k}: {s.index[0].date()} → {s.index[-1].date()}, n={len(s)}, [{s.min():.3f}, {s.max():.3f}], last {s.iloc[-1]:.3f}")
    if k not in ("JPY_OECD_M",): jumps(s, k, 1.0, "pp")
ov = (sr["EONIA"] - sr["ESTR"]).dropna(); p(f"\nEONIA − €STR overlap: {ov.mean()*100:.2f}bp ± {ov.std()*100:.2f}bp (n={len(ov)}) → splice valid" if ov.std() < 0.005 else "")
if ov.std() >= 0.005: flag("FAIL", "EONIA − €STR not constant 8.5bp")
bd = (sr["DTB3"] - sr["DFF"]).dropna(); p(f"DTB3 − DFF since 2000: mean {bd['2000':].mean()*100:.0f}bp (T-bill discount basis + convenience yield)")
jm = sr["JPY_CALL"].resample("MS").mean().to_frame("d").join(sr["JPY_OECD_M"].rename("m"), how="inner").dropna()
p(f"BOJ daily (monthly avg) vs OECD monthly: mean |diff| {(jm.d-jm.m).abs().mean()*100:.1f}bp over {len(jm)} months")
eur = pd.concat([sr["EONIA"][:"2019-09-30"], sr["ESTR"]["2019-10-01":]])
p(f"EUR spliced O/N: {eur.index[0].date()} → {eur.index[-1].date()}, gaps: {(eur.index.to_series().diff().dt.days>7).sum()}")

# ---------- FX ----------
h("6. FX (FRED H.10, noon NY)")
fx = rd("fx/fx_usd.csv"); base(fx, "FX")
for c in fx:
    s = fx[c].dropna(); r = np.log(s).diff().dropna()
    p(f"- {c}: {s.index[0].date()} → {s.index[-1].date()}, [{s.min():.4g}, {s.max():.4g}], max |Δlog| {r.abs().max():.1%} ({r.abs().idxmax().date()}), ann.vol {r.std()*np.sqrt(252):.1%}")
    for dt, v in r[r.abs() > 0.05].items(): flag("WARN", f"{c}: daily move {v:+.1%} on {dt.date()}")
    stale(s, c, 4)
eurjpy = (fx.DEXJPUS * fx.DEXUSEU).dropna(); p(f"- EURJPY (derived): {eurjpy.index[0].date()} → last {eurjpy.iloc[-1]:.2f}")

# ---------- Credit ----------
h("7. Credit")
cr = rd("credit/moodys/moodys_aaa_baa.csv"); base(cr, "Moody's")
q = (cr.DBAA - cr.DAAA).dropna()
p(f"- Baa − Aaa: mean {q.mean()*100:.0f}bp, min {q.min()*100:.0f}bp, max {q.max()*100:.0f}bp ({q.idxmax().date()})")
if (q < 0).any(): flag("FAIL", f"Baa < Aaa on {(q<0).sum()} days")
for c in cr: jumps(cr[c], c, 0.5, "pp")
oas = rd("credit/ice_oas/ice_oas.csv")
p(f"- ICE OAS: {oas.index[0].date()} → {oas.index[-1].date()} (n={len(oas.dropna())}) — **3y history only, reference use**")
if ((oas.BAMLH0A0HYM2 - oas.BAMLC0A0CM).dropna() < 0).any(): flag("FAIL", "HY OAS < IG OAS")

# ---------- VIX ----------
h("8. VIX")
v = rd("volatility/VIXCLS.csv").iloc[:,0].dropna()
p(f"- {v.index[0].date()} → {v.index[-1].date()}, [{v.min():.2f}, {v.max():.2f}] (max {v.idxmax().date()}), median {v.median():.1f}")
if v.min() < 5 or v.max() > 100: flag("FAIL", "VIX out of plausible range")

# ---------- Cross-source sanity ----------
h("9. Cross-source sanity checks")
dy = u["DGS10"].diff()/100
for t, D in [("IEF", 7.5), ("TLT", 17.0), ("SHY", 1.9)]:
    x = pd.concat([etf_ret[t], dy], axis=1, join="inner").dropna()
    b = np.polyfit(x.iloc[:,1], x.iloc[:,0], 1)[0]; cor = x.corr().iloc[0,1]
    p(f"- {t} ret vs ΔDGS10: corr {cor:.2f}, empirical duration {-b:.1f} (ref ≈ {D})")
    if cor > -0.5: flag("WARN", f"{t}: weak link to UST 10Y")
x = pd.concat([etf_ret["SPY"], v.pct_change()], axis=1, join="inner").dropna(); p(f"- SPY ret vs ΔVIX: corr {x.corr().iloc[0,1]:.2f} (expect ≈ −0.7)")
hy = pd.concat([etf_ret["HYG"].resample("ME").apply(lambda r:(1+r).prod()-1), q.resample("ME").last().diff()], axis=1).dropna()
p(f"- HYG monthly ret vs Δ(Baa−Aaa): corr {hy.corr().iloc[0,1]:.2f} (expect negative)")
cal = {k: set(s.index) for k, s in {"US(SPY)": etf_ret["SPY"], "EUR(ECB)": a10.dropna(), "JP(JGB)": j["JGB_10Y"].dropna()}.items()}
lo = pd.Timestamp("2007-05-01"); cal = {k: {d for d in s if d >= lo} for k, s in cal.items()}
p(f"- Calendars since 2007-05: US {len(cal['US(SPY)'])}, EUR {len(cal['EUR(ECB)'])}, JP {len(cal['JP(JGB)'])} days; common {len(set.intersection(*cal.values()))}")

# ---------- Summary ----------
h("Summary")
for lv in ["FAIL", "WARN"]:
    n = sum(1 for l, _ in flags if l == lv); p(f"- {lv}: {n}")
p(f"- INFO: {sum(1 for l,_ in flags if l=='INFO')}")
OUT.write_text("\n".join(L)); print("\n".join(L))
