import argparse, pathlib


def concat(out_dir, stem, domains):
    target = out_dir / f"{stem}.csv"
    with target.open("w", newline="") as sink:
        header = None
        for domain in domains:
            part = out_dir / f"{stem}_{domain}.csv"
            if not part.exists():
                raise SystemExit(f"missing {part}")
            with part.open(newline="") as source:
                first = True
                for line in source:
                    if first:
                        if header is None:
                            header = line
                            sink.write(line)
                        first = False
                        continue
                    sink.write(line)
    print("wrote", target)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=pathlib.Path, required=True)
    ap.add_argument("--domains", default="kit,dv,chng_flu,hosp_cov,respinow,nssp,nhsn,macro_m,rtdsm_q,alfred_w,eia930")
    args = ap.parse_args()
    for stem in ["paired_forecast_losses"]:
        concat(args.dir, stem, args.domains.split(","))
