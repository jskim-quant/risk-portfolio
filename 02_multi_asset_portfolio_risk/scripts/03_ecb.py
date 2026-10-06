import pandas as pd
from common import save

BASE = "https://data-api.ecb.europa.eu/service/data"
MATS = ["3M","1Y","2Y","5Y","7Y","10Y","15Y","20Y","30Y"]

def ecb(flow, key):
    url = f"{BASE}/{flow}/{key}?format=csvdata"
    df = pd.read_csv(url)
    out = df.pivot(index="TIME_PERIOD", columns="KEY", values="OBS_VALUE")
    out.index = pd.to_datetime(out.index); out.index.name = "date"
    return out, url

# G_N_A = AAA, G_N_C = all euro area central gov | SR = zero spot, PY = par yield
for curve in ["G_N_A", "G_N_C"]:
    for typ in ["SR", "PY"]:
        mats = "+".join(f"{typ}_{m}" for m in MATS)
        try:
            df, url = ecb("YC", f"B.U2.EUR.4F.{curve}.SV_C_YM.{mats}")
            save(df, f"rates/ecb/{curve}_{typ}.csv", "ECB Data Portal", url, "%")
        except Exception as e:
            print(f"FAILED {curve}_{typ}: {e}")

for name, flow, key in [("ESTR", "EST", "B.EU000A2X2A25.WT"),
                        ("EONIA", "EON", "D.EONIA_TO.RATE")]:
    try:
        df, url = ecb(flow, key)
        save(df, f"short_rates/{name}.csv", "ECB Data Portal", url, "%")
    except Exception as e:
        print(f"FAILED {name}: {e}  -> search '{name}' on data.ecb.europa.eu")
