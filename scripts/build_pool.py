import os, sys, numpy as np
from modelscope import snapshot_download
import pyarrow.parquet as pq

SPEC = {
    "m4_daily": ("D", 1500, 1), "m4_weekly": ("W", 359, 1), "m4_monthly": ("M", 1500, 1), "m4_quarterly": ("Q", 1000, 1),
    "m4_hourly": ("h", 414, 1), "monash_tourism_monthly": ("M", 366, 1), "monash_tourism_quarterly": ("Q", 427, 1),
    "monash_hospital": ("M", 767, 1), "monash_nn5_weekly": ("W", 111, 1), "dominick": ("W", 1500, 1),
    "monash_traffic": ("h", 400, 1), "uber_tlc_daily": ("D", 262, 1), "monash_pedestrian_counts": ("h", 66, 1),
    "monash_rideshare": ("h", 500, 1), "monash_kdd_cup_2018": ("h", 270, 1), "monash_saugeenday": ("D", 1, 1),
    "monash_weather": ("D", 1000, 1), "wind_farms_daily": ("D", 100, 1), "monash_electricity_weekly": ("W", 321, 1),
    "exchange_rate": ("D", 8, 1), "m5": ("D", 1500, 1), "wiki_daily_100k": ("D", 1500, 5), "taxi_1h": ("h", 500, 1),
    "monash_car_parts": ("M", 1000, 1), "monash_cif_2016": ("M", 72, 1), "monash_m3_monthly": ("M", 1428, 1),
    "monash_m1_monthly": ("M", 617, 1), "monash_m3_quarterly": ("Q", 756, 1), "ushcn_daily": ("D", 500, 1),
    "nn5": ("D", 111, 1), "monash_australian_electricity": ("h", 5, 1),
}
rng = np.random.default_rng(0)
series, freqs, names = [], [], []
for cfg, (fr, nmax, nsh) in SPEC.items():
    try:
        root = snapshot_download("autogluon/chronos_datasets", repo_type="dataset", allow_patterns=[f"{cfg}/train-00000-of-{nsh:05d}.parquet"],
                                 cache_dir="data/chronos_datasets")
        f = f"{root}/{cfg}/train-00000-of-{nsh:05d}.parquet"
        tab = pq.read_table(f)
    except Exception as e:
        print("FAILED", cfg, e, flush=True); continue
    cols = tab.column_names
    tcols = [c for c in cols if c not in ("id", "timestamp")]
    n = tab.num_rows
    idx = rng.choice(n, size=min(n, nmax), replace=False)
    cnt = 0
    for i in idx:
        for tc in tcols[:1]:
            v = tab.column(tc)[int(i)].as_py()
            if v is None or not isinstance(v, list) or (len(v) and isinstance(v[0], list)):
                continue
            x = np.asarray(v, dtype=np.float32)
            if len(x) > 4000:
                st = rng.integers(0, len(x) - 4000 + 1); x = x[st:st + 4000]
            if np.isfinite(x).sum() < 60:
                continue
            series.append(x); freqs.append(fr); names.append(cfg); cnt += 1
    print(cfg, "cols", tcols[:3], "added", cnt, "of", n, flush=True)
np.savez_compressed("data/pool.npz", series=np.array(series, dtype=object), freq=np.array(freqs), name=np.array(names))
print("POOL_DONE", len(series))
