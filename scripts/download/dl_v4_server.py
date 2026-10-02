import os, time, subprocess, gzip, shutil, threading, queue
OUT = "data/raw/delphi_v4"
os.makedirs(OUT, exist_ok=True)
LOG = open(OUT + "/download.log", "a")
lock = threading.Lock()
def log(*a):
    with lock:
        print(time.strftime("%H:%M:%S"), *a, file=LOG, flush=True)

STATES = "al ak az ar ca co ct de dc fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy".split()
V4 = [
    ("doctor-visits", "smoothed_adj_cli", "20200501", "20260919", STATES, None),
    ("chng", "smoothed_adj_outpatient_flu", "20200501", "20240301", STATES, None),
    ("hospital-admissions", "smoothed_adj_covid19_from_claims", "20200501", "20260919",
     "ca tx fl ny pa il oh ga nc mi nj va".split(), "year"),
]
q = queue.Queue()
for src, sig, t0, t1, geos, chunk in V4:
    for g in geos:
        if chunk is None:
            spans = [(t0, t1)]
        else:
            years = range(int(t0[:4]), int(t1[:4]) + 1)
            spans = [(max(t0, f"{y}0101"), min(t1, f"{y}1231")) for y in years]
        for a, b in spans:
            suffix = "" if chunk is None else f"__{a[:4]}"
            out = f"{OUT}/{src}__{sig}__{g}{suffix}.csv"
            if os.path.exists(out + ".gz"):
                continue

            import datetime as _dt
            b_iss = min(t1, (_dt.datetime.strptime(b, "%Y%m%d") + _dt.timedelta(days=500)).strftime("%Y%m%d"))
            url = ("https://api.delphi.cmu.edu/epidata/covidcast/?data_source=" + src + "&signal=" + sig +
                   "&time_type=day&geo_type=state&time_values=" + a + "-" + b + "&geo_value=" + g +
                   "&issues=" + a + "-" + b_iss + "&format=csv&fields=time_value,issue,value")
            q.put((url, out))
log("pending", q.qsize())
rate_limited_until = [0.0]

def worker(wid):
    while True:
        try:
            url, out = q.get_nowait()
        except queue.Empty:
            return
        for attempt in range(8):
            now = time.time()
            if now < rate_limited_until[0]:
                time.sleep(rate_limited_until[0] - now + 5)
            t0 = time.time()
            r = subprocess.run(["curl", "-s", "--compressed", "--max-time", "400", "-o", out + f".part{wid}", "-w", "%{http_code}", url], capture_output=True, text=True)
            code = r.stdout.strip(); dt = time.time() - t0
            p = out + f".part{wid}"
            size = os.path.getsize(p) if os.path.exists(p) else 0
            ok = False
            if code == "200" and size > 100:
                with open(p, "rb") as f:
                    f.seek(-1, 2); last = f.read(1)
                if last == b"\n":
                    with open(p, "rb") as f, gzip.open(out + ".gz", "wb", 6) as gz:
                        shutil.copyfileobj(f, gz)
                    ok = True
            if os.path.exists(p):
                os.remove(p)
            log("OK" if ok else "FAIL", os.path.basename(out), code, size, f"{dt:.0f}s", f"w{wid}", f"try{attempt}")
            if ok:
                break
            if code == "429":
                rate_limited_until[0] = time.time() + 900
                log("rate limited; sleeping 15 min")
            else:
                time.sleep(20 * (attempt + 1))
        time.sleep(2)

threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
for t in threads: t.start()
for t in threads: t.join()
log("ALL DONE")
