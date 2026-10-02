import argparse, csv, json, pathlib
import numpy as np

DOMAINS = ['kit', 'dv', 'chng_flu', 'hosp_cov', 'respinow', 'nssp', 'nhsn',
           'macro_m', 'rtdsm_q', 'alfred_w', 'eia930']
FIELDS = ['domain', 'sid', 'origin', 'origin_date', 'scale', 'above_1pct_scale',
          'recast_mase_sum', 'recast_count', 'cl_mase_sum', 'cl_count',
          'latest_report_mase_sum', 'latest_report_count']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=pathlib.Path, required=True)
    ap.add_argument('--results', type=pathlib.Path, default=pathlib.Path('results'))
    args = ap.parse_args()
    out = args.dir
    out.mkdir(parents=True, exist_ok=True)
    rr = args.results / 'rrbench'
    aggregate = json.load((args.results / 'agg_v12_final.json').open())
    checks = []
    total = 0
    with (out / 'paired_nowcast_bootstrap_input.csv').open('w', newline='') as sink:
        writer = csv.DictWriter(sink, FIELDS)
        writer.writeheader()
        for domain in DOMAINS:
            with np.load(rr / f'v12_final__{domain}.npz', allow_pickle=True) as data:
                index = {(str(s), int(t)): i for i, (s, t) in enumerate(zip(data['index_sid'], data['index_t']))}
                xn, yn, scale = data['XN'], data['YN'], data['S']
            mask = np.isfinite(xn) & np.isfinite(yn)
            error = np.abs(xn - yn) / scale[:, None]
            sums = np.where(mask, error, 0).sum(1)
            counts = mask.sum(1)
            observed_sum, observed_count = 0.0, 0
            with (out / f'paired_nowcast_{domain}.csv').open(newline='') as source:
                for row in csv.DictReader(source):
                    i = index[(row['sid'], int(row['origin']))]
                    result = {field: row[field] for field in FIELDS if field in row}
                    result.update(latest_report_mase_sum=float(sums[i]), latest_report_count=int(counts[i]))
                    writer.writerow(result)
                    observed_sum += float(sums[i])
                    observed_count += int(counts[i])
                    total += 1
            observed = observed_sum / observed_count
            expected = aggregate[domain]['naive_now']['MASE']
            checks.append(dict(domain=domain, method='latest_report', task='nowcast', metric='MASE',
                               recomputed=observed, stored=expected,
                               absolute_difference=abs(observed - expected), n_cells=observed_count))
    with (out / 'latest_report_recompute_checks.csv').open('w', newline='') as sink:
        writer = csv.DictWriter(sink, list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    print('paired_nowcast_bootstrap_input.csv', total, 'x', len(FIELDS))


if __name__ == '__main__':
    main()
