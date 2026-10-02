import os, glob, numpy as np, pandas as pd
D = "data/basesignals"
series, freqs, names = [], [], []

def add(x, fr, name, minlen=80):
    x = np.asarray(x, dtype=np.float32)
    if np.isfinite(x).sum() >= minlen and np.nanstd(x) > 0:
        series.append(x); freqs.append(fr); names.append(name)

def add_hourly_frame(df, name, max_cols=400):

    df = df.select_dtypes("number")
    cols = list(df.columns)[:max_cols]
    for c in cols:
        s = df[c].astype(float)
        add(s.to_numpy(), "h", name)
        d = s.resample("D").mean(); add(d.to_numpy(), "D", name)
        if len(d) > 200:
            w = s.resample("W").mean(); add(w.to_numpy(), "W", name)
        if len(d) > 800:
            m = s.resample("MS").mean(); add(m.to_numpy(), "M", name, 40)


for f in glob.glob(f"{D}/ett/extracted/ETT/ETT*.csv"):
    df = pd.read_csv(f, parse_dates=["date"]).set_index("date")
    if "m" in os.path.basename(f):
        df = df.resample("h").mean()
    add_hourly_frame(df, "ett")

try:
    hp = pd.read_csv(f"{D}/electricity/extracted/household_power_consumption.txt", sep=";", na_values="?", low_memory=False)
    hp["ts"] = pd.to_datetime(hp["Date"] + " " + hp["Time"], format="%d/%m/%Y %H:%M:%S")
    hp = hp.set_index("ts").drop(columns=["Date", "Time"]).apply(pd.to_numeric, errors="coerce").resample("h").mean()
    add_hourly_frame(hp, "household")
except Exception as e:
    print("household failed", e)

for f in glob.glob(f"{D}/beijing_uci/PRSA_Data_20130301-20170228/*.csv"):
    df = pd.read_csv(f)
    df["ts"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.set_index("ts")[["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "WSPM"]]
    add_hourly_frame(df, "prsa")

try:
    tx = pd.read_csv(f"{D}/nyc_taxi_nab/nyc_taxi.csv", parse_dates=["timestamp"]).set_index("timestamp").resample("h").sum()
    add_hourly_frame(tx, "nyc_taxi")
except Exception as e:
    print("taxi failed", e)

try:
    ne = pd.read_csv(f"{D}/new_energy/电厂站功率.csv")
    ne.columns = ["ts", "id", "p"]
    ne["ts"] = pd.to_datetime(ne["ts"])
    piv = ne.pivot_table(index="ts", columns="id", values="p").resample("h").mean()
    add_hourly_frame(piv, "new_energy", 200)
except Exception as e:
    print("new_energy failed", e)

try:
    ac = pd.read_csv(f"{D}/air_city/air_city_hour.csv", usecols=["release_date", "city_code", "PM25", "AQI", "NO2"])
    ac["ts"] = pd.to_datetime(ac["release_date"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    for col in ["PM25", "AQI"]:
        piv = ac.pivot_table(index="ts", columns="city_code", values=col, aggfunc="mean").resample("h").mean()
        add_hourly_frame(piv, f"air_city_{col}", 300)
except Exception as e:
    print("air_city failed", e)

try:
    f = glob.glob(f"{D}/air_city_day/*.csv")[0]
    ad = pd.read_csv(f)
    print("air_city_day cols", list(ad.columns)[:12])
    dcol = [c for c in ad.columns if "date" in c.lower() or "日期" in c][0]
    ccol = [c for c in ad.columns if "city" in c.lower() or "城市" in c][0]
    ad[dcol] = pd.to_datetime(ad[dcol], errors="coerce")
    for col in [c for c in ad.columns if c.upper().startswith(("PM", "AQI", "SO2", "NO2", "O3", "CO"))][:4]:
        piv = ad.pivot_table(index=dcol, columns=ccol, values=col, aggfunc="mean").resample("D").mean()
        for c in list(piv.columns)[:400]:
            s = piv[c]
            add(s.to_numpy(), "D", "air_city_day")
            add(s.resample("W").mean().to_numpy(), "W", "air_city_day")
except Exception as e:
    print("air_city_day failed", e)

from collections import Counter
print(Counter(freqs), Counter(names))
np.savez_compressed("data/pool_local.npz", series=np.array(series, dtype=object), freq=np.array(freqs), name=np.array(names))
print("POOL_DONE", len(series))
