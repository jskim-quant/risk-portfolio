import io, requests, pandas as pd
from common import save, save_bytes

URLS = {
  "hist": "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv",
  "cur":  "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv",
}
frames = []
for k, url in URLS.items():
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    save_bytes(r.content, f"rates/jgb/original_{k}.csv")
    df = pd.read_csv(io.BytesIO(r.content), skiprows=1, na_values="-", encoding="cp932")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d", errors="coerce")
    df = df.dropna(subset=["Date"])
    frames.append(df.set_index("Date"))

jgb = pd.concat(frames).sort_index()
jgb = jgb[~jgb.index.duplicated(keep="last")]
jgb.index.name = "date"
jgb.columns = [f"JGB_{c}" for c in jgb.columns]
save(jgb, "rates/jgb/jgb_curve.csv", "Japan MoF", URLS["hist"], "%")
