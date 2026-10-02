import argparse, gzip, json, os, time, urllib.parse, urllib.request

API = "https://api.delphi.cmu.edu/epidata/covidcast/"
SOURCES = {
    "nhsn": ["confirmed_admissions_covid_ew", "confirmed_admissions_flu_ew", "confirmed_admissions_rsv_ew"],
    "nssp": ["pct_ed_visits_covid", "pct_ed_visits_influenza", "pct_ed_visits_rsv"],
}
STATES = ("al ak az ar ca co ct de dc fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny "
          "nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy").split()


def week_start(value):
    year, week = int(value) // 100, int(value) % 100
    import datetime as dt
    return dt.date.fromisocalendar(year, week, 1).isoformat()


def week_end(value):
    import datetime as dt
    year, week = int(value) // 100, int(value) % 100
    return dt.date.fromisocalendar(year, week, 7).isoformat()


def fetch(url, attempts=6, timeout=600):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            print("  retry", attempt, exc, flush=True)
            time.sleep(20 * (attempt + 1))
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/delphi_v5")
    ap.add_argument("--first", default="202232", help="first reference week, YYYYWW")
    ap.add_argument("--last", default="202637", help="last reference week, YYYYWW")
    ap.add_argument("--sources", default="nhsn,nssp")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for source in args.sources.split(","):
        for signal in SOURCES[source]:
            out = f"{args.out}/{source}__{signal}__state.csv.gz"
            if os.path.exists(out):
                print("exists", out, flush=True)
                continue
            rows = ["signal,report_time,geo_type,geo_value,fill_method,reference_time,value"]
            records = []
            for state in STATES:
                query = {"data_source": source, "signal": signal, "time_type": "week", "geo_type": "state",
                         "time_values": f"{args.first}-{args.last}", "geo_value": state,
                         "issues": f"{args.first}-{args.last}"}
                body = fetch(API + "?" + urllib.parse.urlencode(query))
                print(source, signal, state, len(body), flush=True)
                records.extend(body)
                time.sleep(0.4)
            for rec in records:
                value = rec.get("value")
                if value is None:
                    continue
                rows.append(f"{signal},{week_start(rec['issue'])},state,{rec['geo_value']},source,"
                            f"{week_end(rec['time_value'])},{value}")
            with gzip.open(out, "wt", encoding="utf-8") as handle:
                handle.write("\n".join(rows) + "\n")
            print("saved", out, len(rows) - 1, flush=True)


if __name__ == "__main__":
    main()
