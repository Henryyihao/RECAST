import re, io, os, sys, time, subprocess
import pandas as pd
OUT = "data/raw/alfred"
os.makedirs(OUT, exist_ok=True)
SERIES = sys.argv[1:] or ["ICSA", "CCSA", "RSAFS", "RSXFS", "UNRATE", "PAYEMS", "INDPRO", "HOUST", "TOTALSA", "DGORDER", "CPIAUCSL", "PERMIT", "RRSFS"]
BATCH = 12

def curl(url, timeout=120):
    for a in range(4):
        r = subprocess.run(["curl", "-s", "-L", "--compressed", "--max-time", str(timeout), url], capture_output=True)
        if r.returncode == 0 and len(r.stdout) > 50:
            return r.stdout
        time.sleep(10 * (a + 1))
    return b""

for sid in SERIES:
    outp = f"{OUT}/{sid}.parquet"
    if os.path.exists(outp):
        print(sid, "exists"); continue
    page = curl(f"https://alfred.stlouisfed.org/series/downloaddata?seid={sid}").decode("utf-8", "replace")
    m = re.search(r'name="form\[selected_vintage_dates\]\[\]".*?</select>', page, flags=re.S)
    if not m:
        print(sid, "no vintage select found"); continue
    vints = re.findall(r'value="(\d{4}-\d{2}-\d{2})"', m.group(0))
    vints = sorted(set(vints))
    print(sid, "vintages", len(vints), vints[:1], vints[-1:], flush=True)
    frames = []
    for i in range(0, len(vints), BATCH):
        b = vints[i:i + BATCH]
        url = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=" + ",".join([sid] * len(b)) + "&vintage_date=" + ",".join(b)
        raw = curl(url, timeout=180)
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            print("  parse fail", i, e, raw[:100]); continue
        if "observation_date" not in df.columns:
            print("  bad columns", df.columns[:3].tolist()); continue
        long = df.melt(id_vars="observation_date", var_name="col", value_name="value")
        long["report_time"] = pd.to_datetime(long["col"].str.extract(r"_(\d{8})$")[0], format="%Y%m%d")
        long = long.dropna(subset=["report_time"])
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long = long.dropna(subset=["value"])
        long["reference_time"] = pd.to_datetime(long["observation_date"])
        frames.append(long[["reference_time", "report_time", "value"]])
        print(f"  batch {i//BATCH+1}/{(len(vints)+BATCH-1)//BATCH} rows {len(long)}", flush=True)
        time.sleep(1)
    if frames:
        allf = pd.concat(frames, ignore_index=True)
        allf["series_id"] = sid
        allf.to_parquet(outp, index=False)
        print(sid, "saved", len(allf), flush=True)
print("ALL DONE")
