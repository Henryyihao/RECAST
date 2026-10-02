from __future__ import annotations
import glob, gzip, io, json, os, re
import numpy as np
import pandas as pd
from .vintage import (age_view_from_changes, age_view_from_snapshots, age_view_from_triangle,
                      rolling_sum_age_view, revision_profile)

RAW = "data/raw"
OUT = "data/bundles"


def save_bundle(b: dict):
    os.makedirs(OUT, exist_ok=True)
    arrays = {}
    meta = {"name": b["name"], "freq": b["freq"], "L": int(b["L"]), "series": [], "notes": b.get("notes", "")}
    for i, (sid, s) in enumerate(b["series"].items()):
        arrays[f"A_{i}"] = s["A"].astype(np.float32)
        arrays[f"time_{i}"] = np.asarray(s["time"]).astype("datetime64[ns]").astype("int64")
        arrays[f"R_{i}"] = np.asarray(s["R"], dtype=np.float32)
        meta["series"].append({"idx": i, "id": sid, "T": int(s["A"].shape[0]), **{k: v for k, v in s.items() if k not in ("A", "time", "R")}})
    np.savez_compressed(f"{OUT}/{b['name']}.npz", **arrays)
    json.dump(meta, open(f"{OUT}/{b['name']}.json", "w"), indent=1)
    print("saved", b["name"], len(meta["series"]), "series")


def load_bundle(name: str) -> dict:
    meta = json.load(open(f"{OUT}/{name}.json"))
    z = np.load(f"{OUT}/{name}.npz")
    series = {}
    for s in meta["series"]:
        i = s["idx"]
        series[s["id"]] = {"A": z[f"A_{i}"], "R": z[f"R_{i}"].astype(np.float64), "time": z[f"time_{i}"].astype("datetime64[ns]"), **{k: v for k, v in s.items() if k not in ("idx", "id", "T")}}
    return {"name": meta["name"], "freq": meta["freq"], "L": meta["L"], "series": series, "notes": meta.get("notes", "")}


def build_delphi_v4(source: str, signal: str, name: str, L: int = 75, min_issue_gap_days: int = 0):
    files = sorted(glob.glob(f"{RAW}/delphi_v4/{source}__{signal}__*.csv.gz"))
    by_state = {}
    for f in files:
        m = re.search(r"__([a-z]{2})(?:__(\d{4}))?\.csv\.gz$", f)
        if not m:
            continue
        by_state.setdefault(m.group(1), []).append(f)
    series = {}
    for st, fl in sorted(by_state.items()):
        d = pd.concat([pd.read_csv(f) for f in fl], ignore_index=True)
        d = d.drop_duplicates(["time_value", "issue"], keep="last")
        if d.empty:
            continue
        d["ref"] = pd.to_datetime(d["time_value"].astype(str), format="%Y%m%d")
        d["rep"] = pd.to_datetime(d["issue"].astype(str), format="%Y%m%d")
        t0 = d["ref"].min(); t1 = d["ref"].max()
        grid = pd.date_range(t0, t1, freq="D")
        ref_idx = ((d["ref"] - t0).dt.days).to_numpy()
        rep_idx = ((d["rep"] - t0).dt.days).to_numpy()
        A, R = age_view_from_changes(ref_idx, rep_idx, d["value"].to_numpy(), len(grid), L)
        series[st] = {"A": A, "R": R, "time": grid.to_numpy()}
    b = {"name": name, "freq": "D", "L": L, "series": series, "notes": f"Delphi COVIDcast V4 {source}/{signal}, state level, daily issues"}
    save_bundle(b)
    return b


def build_delphi_v5(source: str, signal: str, name: str, L: int = 10, min_ref: str | None = None):
    f = f"{RAW}/delphi_v5/{source}__{signal}__state.csv.gz"
    d = pd.read_csv(f, usecols=["report_time", "geo_value", "reference_time", "value"])
    d["report_time"] = pd.to_datetime(d["report_time"])
    d["reference_time"] = pd.to_datetime(d["reference_time"])
    if min_ref:
        d = d[d["reference_time"] >= pd.Timestamp(min_ref)]
    series = {}
    for geo, g in d.groupby("geo_value"):
        piv = g.pivot_table(index="reference_time", columns="report_time", values="value", aggfunc="last")

        t0, t1 = piv.index.min(), piv.index.max()
        grid = pd.date_range(t0, t1, freq="7D")
        piv = piv.reindex(grid)
        reps = piv.columns
        rep_grid_idx = np.floor(((reps - t0) / pd.Timedelta(days=7)).to_numpy()).astype(int)
        A, R = age_view_from_snapshots(piv.to_numpy(dtype=np.float64), rep_grid_idx, L)
        series[geo] = {"A": A, "R": R, "time": grid.to_numpy()}
    b = {"name": name, "freq": "W", "L": L, "series": series, "notes": f"Delphi V5 {source}/{signal}, state level, weekly snapshots"}
    save_bundle(b)
    return b


def build_kit(name: str = "kit_hosp7", L: int = 80, roll: int = 7, age_groups=("00+",), all_states: bool = True):
    d = pd.read_csv(f"{RAW}/kit/COVID-19_hospitalizations_preprocessed.csv")
    d["date"] = pd.to_datetime(d["date"])
    inc_cols = [c for c in d.columns if re.match(r"value_\d+d$", c)]
    inc_cols = sorted(inc_cols, key=lambda c: int(re.search(r"(\d+)", c).group(1)))
    series = {}
    locs = sorted(d["location"].unique()) if all_states else ["DE"]
    for loc in locs:
        for ag in age_groups:
            g = d[(d["location"] == loc) & (d["age_group"] == ag)].sort_values("date")
            if g.empty:
                continue
            grid = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
            g = g.set_index("date").reindex(grid)
            inc = g[inc_cols].to_numpy(dtype=np.float64)
            A, R = age_view_from_triangle(inc, L)
            if roll > 1:
                A, R = rolling_sum_age_view(A, R, roll, L)
            series[f"{loc}|{ag}"] = {"A": A, "R": R, "time": grid.to_numpy()}
    b = {"name": name, "freq": "D", "L": L, "series": series, "notes": f"KIT hospitalization nowcast hub reporting triangle; {roll}-day rolling sum of daily admissions"}
    save_bundle(b)
    return b


def build_respinow(name: str = "respinow", L: int = 10, sources=("icosari-sari", "survstat-influenza", "survstat-rsv", "agi-are", "nrz-influenza", "nrz-rsv")):
    series = {}
    for src in sources:
        f = f"{RAW}/respinow/reporting_triangle-{src}.csv"
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        d["date"] = pd.to_datetime(d["date"])
        inc_cols = [c for c in d.columns if re.match(r"value_-?\d+w$", c)]
        inc_cols = sorted(inc_cols, key=lambda c: int(re.search(r"(-?\d+)", c).group(1)))
        for (loc, ag), g in d.groupby(["location", "age_group"]):
            g = g.sort_values("date")
            grid = pd.date_range(g["date"].min(), g["date"].max(), freq="7D")
            g = g.set_index("date").reindex(grid)
            inc = g[inc_cols].to_numpy(dtype=np.float64)
            if inc_cols[0] == "value_-1w":
                inc = np.concatenate([np.nansum(inc[:, :2], axis=1, keepdims=True), inc[:, 2:]], axis=1)
            A, R = age_view_from_triangle(inc, L)
            if np.nanmean(A[:, L]) < 20 or np.nanmedian(A[:, L]) < 5:
                continue
            series[f"{src}|{loc}|{ag}"] = {"A": A, "R": R, "time": grid.to_numpy()}
    b = {"name": name, "freq": "W", "L": L, "series": series, "notes": "RESPINOW-Hub weekly reporting triangles (Germany)"}
    save_bundle(b)
    return b


def _rtdsm_vintage_date(col: str, freq: str):
    m = re.search(r"(\d{2})([MQ])(\d{1,2})$", col)
    yy, _, p = int(m.group(1)), m.group(2), int(m.group(3))
    year = 1900 + yy if yy >= 47 else 2000 + yy
    if freq == "M":
        return pd.Period(year=year, month=p, freq="M")
    return pd.Period(year=year, quarter=p, freq="Q")


def build_rtdsm(freq: str = "M", name: str | None = None, L: int | None = None, log_transform_ids=()):
    name = name or f"rtdsm_{freq.lower()}"
    L = L if L is not None else (12 if freq == "M" else 8)
    pat = "mvmd" if freq == "M" else "qvqd"
    series = {}
    for f in sorted(glob.glob(f"{RAW}/rtdsm/*{pat}.xlsx")):
        sid = os.path.basename(f).replace(f"{pat}.xlsx", "").upper()
        df = pd.read_excel(f)
        df = df.rename(columns={df.columns[0]: "DATE"})
        if freq == "M":
            ref = pd.PeriodIndex([pd.Period(year=int(s[:4]), month=int(s[5:7]), freq="M") for s in df["DATE"].astype(str)], freq="M")
        else:
            ref = pd.PeriodIndex([pd.Period(year=int(s[:4]), quarter=int(s[-1]), freq="Q") for s in df["DATE"].astype(str)], freq="Q")
        vcols = [c for c in df.columns if c != "DATE"]
        vdates = [_rtdsm_vintage_date(c, freq) for c in vcols]
        order = np.argsort([v.ordinal for v in vdates])
        vcols = [vcols[i] for i in order]; vdates = [vdates[i] for i in order]
        mat = df[vcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

        t0 = ref[0]
        rep_grid_idx = np.asarray([v.ordinal - t0.ordinal for v in vdates])
        A, R = age_view_from_snapshots(mat, rep_grid_idx, L)
        time = ref.to_timestamp(how="end").normalize().to_numpy()
        series[sid] = {"A": A, "R": R, "time": time}
    b = {"name": name, "freq": freq, "L": L, "series": series, "notes": "Philadelphia Fed Real-Time Data Set for Macroeconomists"}
    save_bundle(b)
    return b


def build_alfred(name: str, ids, freq: str, L: int):
    series = {}
    for sid in ids:
        f = f"{RAW}/alfred/{sid}.parquet"
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f)
        d = d.dropna(subset=["value"])
        piv = d.pivot_table(index="reference_time", columns="report_time", values="value", aggfunc="last")
        step = {"W": pd.Timedelta(days=7), "M": None}[freq]
        if freq == "W":
            t0 = piv.index.min(); grid = pd.date_range(t0, piv.index.max(), freq="7D")
            piv = piv.reindex(grid)
            rep_grid_idx = np.floor(((piv.columns - t0) / step).to_numpy()).astype(int)
            time = grid.to_numpy()
        else:
            per = piv.index.to_period("M"); t0 = per.min()
            grid = pd.period_range(t0, per.max(), freq="M")
            piv.index = per; piv = piv.reindex(grid)
            rep_grid_idx = np.asarray([(pd.Period(c, freq="M").ordinal - t0.ordinal) for c in piv.columns])
            time = grid.to_timestamp(how="end").normalize().to_numpy()
        A, R = age_view_from_snapshots(piv.to_numpy(dtype=np.float64), rep_grid_idx, L)
        series[sid] = {"A": A, "R": R, "time": time}
    b = {"name": name, "freq": freq, "L": L, "series": series, "notes": "ALFRED vintages via alfredgraph"}
    save_bundle(b)
    return b


def build_eia930(name: str = "eia930", L: int = 72, min_hours: int = 20000, max_bas: int | None = None):
    frames = []
    for f in sorted(glob.glob(f"{RAW}/eia930/EIA930_BALANCE_*.csv")):
        d = pd.read_csv(f, usecols=["Balancing Authority", "UTC Time at End of Hour", "Demand (MW)", "Demand (MW) (Adjusted)"], thousands=",", low_memory=False)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d.columns = ["ba", "utc", "raw", "adj"]
    d["utc"] = pd.to_datetime(d["utc"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    d["raw"] = pd.to_numeric(d["raw"], errors="coerce"); d["adj"] = pd.to_numeric(d["adj"], errors="coerce")
    d = d.dropna(subset=["utc"])
    series = {}
    for ba, g in d.groupby("ba"):
        g = g.drop_duplicates("utc").set_index("utc").sort_index()
        if len(g) < min_hours or g["adj"].notna().mean() < 0.9:
            continue
        grid = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.reindex(grid)
        raw = g["raw"].to_numpy(dtype=np.float64); adj = g["adj"].to_numpy(dtype=np.float64)
        adj = pd.Series(adj).interpolate(limit=6).to_numpy()
        raw = np.where(np.isfinite(raw), raw, np.nan)
        A = np.empty((len(grid), L + 1), dtype=np.float32)
        A[:, :L] = raw[:, None]
        A[:, L] = adj
        R = np.arange(len(grid), dtype=float); R[~np.isfinite(raw)] = np.nan
        series[ba] = {"A": A, "R": R, "time": grid.to_numpy()}
        if max_bas and len(series) >= max_bas:
            break
    b = {"name": name, "freq": "h", "L": L, "series": series, "notes": "EIA-930 hourly demand: raw report (provisional) vs adjusted (final); two-version structure"}
    save_bundle(b)
    return b


def bundle_summary(b: dict, max_series: int = 3):
    L = b["L"]
    rows = []
    for sid, s in list(b["series"].items()):
        prof = revision_profile(s["A"], L)
        rows.append({"id": sid, "T": s["A"].shape[0], "first_age": float(np.nanmedian(s["R"] - np.arange(len(s["R"])))),
                     "medrel_a0": prof[0, 0], "medrel_a1": prof[1, 0], "medrel_mid": prof[L // 4, 0], "meanrel_a0": prof[0, 1]})
    df = pd.DataFrame(rows)
    print(b["name"], "freq", b["freq"], "L", L, "n_series", len(rows))
    print(df.describe().loc[["mean", "50%"]].to_string())
    return df


def merge_bundles(name: str, parts: list, prefix: bool = True, notes: str = ""):

    series = {}
    freq = L = None
    for p in parts:
        b = load_bundle(p)
        freq = freq or b["freq"]; L = L or b["L"]
        assert b["freq"] == freq and b["L"] == L, (p, b["freq"], b["L"])
        for sid, s_ in b["series"].items():
            series[(f"{p}|{sid}" if prefix else sid)] = s_
    b = {"name": name, "freq": freq, "L": L, "series": series, "notes": notes or ("merged: " + ", ".join(parts))}
    save_bundle(b)
    return b
