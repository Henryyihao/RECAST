from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path('.')
RESULTS = ROOT / 'results'
RR = RESULTS / 'rrbench'
FPD = Path('data/outputside')
DOMAINS = ['kit', 'dv', 'chng_flu', 'hosp_cov', 'respinow', 'nssp', 'nhsn',
           'macro_m', 'rtdsm_q', 'alfred_w', 'eia930']
METRICS = ['MASE', 'WQL', 'CRPS_s', 'COV80', 'WIDTH80', 'PCE', 'n_cells']
LEVELS = np.arange(.1, 1., .1)
EXAMPLES = [
    ('kit', 'kit_hosp7|DE|00+', 'DE-Hosp, national', 120),
    ('dv', 'ny', 'US-CLI, New York', 120),
    ('chng_flu', 'tx', 'US-Flu, Texas', 120),
    ('respinow', 'agi-are|DE-NW|00+', 'DE-RESP, North Rhine-Westphalia', 60),
]


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def value(x):
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return ''
    return x.item() if isinstance(x, np.generic) else x


def write_csv(out, name, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: value(row.get(k)) for k in fields})
    print(name, len(rows), 'x', len(fields), flush=True)
    return {'file': name, 'rows': len(rows), 'columns': fields}


def metadata(domain, tag='v12_final'):
    source = RR / f'{tag}__{domain}.json'
    d = read_json(source if source.exists() else RR / f'v12_final__{domain}.json')
    return {k: d.get(k) for k in ['freq', 'L', 'C', 'H', 'N', 'N_model']}


def metric_exports(out):
    artifacts = []
    mapping = {
        'main': ('agg_v12_final.json', 'chronos2'),
        'bolt_s': ('agg_bolt_s.json', 'bolt_s'),
        'bolt_b': ('agg_bolt_b.json', 'bolt_b'),
        'toto': ('agg_toto.json', 'toto'),
        'timemoe': ('agg_timemoe.json', 'timemoe'),
        'ablation': ('agg_abl2.json', 'chronos2'),
        'sensitivity': ('agg_sens.json', 'chronos2'),
        'bayesian': ('agg_bayes.json', 'chronos2'),
    }
    all_rows, all_horizons = [], []
    for comparison, (filename, backbone) in mapping.items():
        source = RESULTS / filename
        aggregate = read_json(source)
        rows, horizons = [], []
        for domain, data in aggregate.items():
            default_tag = f'{backbone}_v3' if comparison in ['bolt_s', 'bolt_b', 'toto', 'timemoe'] else 'v12_final'
            md = metadata(domain, default_tag)
            common = dict(comparison=comparison, backbone=backbone, domain=domain,
                          freq=md['freq'], L=md['L'], C=md['C'], H=md['H'],
                          N_score=md['N'], N_model=md['N_model'],
                          n_common=data.get('n_common'), n_eval=data.get('n_eval'),
                          source=str(source))
            for method, entry in data.items():
                if not isinstance(entry, dict) or method in ['tests']:
                    continue
                for task in ['forecast', 'nowcast']:
                    if task not in entry:
                        continue
                    method_md = metadata(domain, method)
                    method_common = dict(common, N_model=method_md['N_model'] or md['N_model'])
                    rows.append(dict(method_common, method=method, task=task,
                                     **{m: entry[task].get(m) for m in METRICS}))
                    hentry = entry.get(task + '_h', {})
                    length = max([len(v) for v in hentry.values()] or [0])
                    for j in range(length):
                        row = dict(method_common, method=method, task=task, horizon=j + 1,
                                   report_age=md['N'] - j - 1 if task == 'nowcast' else None)
                        for metric, key in [('MASE', 'MASE_h'), ('CRPS_s', 'CRPS_h'), ('COV80', 'COV80_h')]:
                            row[metric] = hentry.get(key, [None] * length)[j]
                        horizons.append(row)
            if 'naive_now' in data:
                rows.append(dict(common, method='latest_report', task='nowcast',
                                 **{m: data['naive_now'].get(m) for m in METRICS}))
        artifacts.append(write_csv(out, comparison + '_metrics.csv', rows))
        artifacts.append(write_csv(out, comparison + '_horizons.csv', horizons))
        all_rows.extend(rows)
        all_horizons.extend(horizons)
    artifacts.append(write_csv(out, 'all_comparison_metrics.csv', all_rows))
    artifacts.append(write_csv(out, 'all_comparison_horizons.csv', all_horizons))
    artifacts.append(write_csv(out, 'backbone_metrics.csv',
                               [r for r in all_rows if r['comparison'] in ['main', 'bolt_s', 'bolt_b', 'toto', 'timemoe']]))
    return artifacts


def mechanism_exports(out):
    source = RESULTS / 'mech/mech_v12_main.json'
    data = read_json(source)
    artifacts = [write_csv(out, 'mechanism_overall.csv',
                           [dict(method=m, source=str(source), **v) for m, v in data['M1'].items()])]
    rows = [dict(method=m, report_age=i, MASE=v, n_per_age=300, source=str(source))
            for m, vals in data['M1_by_age'].items() for i, v in enumerate(vals)]
    artifacts.append(write_csv(out, 'mechanism_age.csv', rows))
    rows = [dict(method=m, steps_since_switch=i, MASE=v, n_series=150,
                 n_cells=150 * 40, source=str(source))
            for m, vals in data['M2'].items() for i, v in enumerate(vals) if v is not None]
    artifacts.append(write_csv(out, 'mechanism_switch.csv', rows))
    rows = [dict(probe=k.removeprefix('r2_'), R2=v, n=data['M3']['n'],
                 target='log(mean_delay+0.001)', outer_folds=5,
                 inner_rule='RidgeCV default leave-one-out', source=str(source))
            for k, v in data['M3'].items() if k.startswith('r2_')]
    artifacts.append(write_csv(out, 'mechanism_probe.csv', rows))
    return artifacts


def dataeff_exports(out):
    rows = []
    for domain in ['kit', 'dv']:
        for C in [96, 128, 192, 256]:
            tag = f'v12_C{C}' if C != 256 else 'v12_final'
            btag = f'baselines_C{C}' if C != 256 else 'baselines'
            for tag, methods in [(tag, [('RECAST', 'nowcast'), ('RECAST', 'forecast')]),
                                 (btag, [('chain_ladder', 'nowcast_cl'), ('CL_pipeline', 'forecast_2s_point')])]:
                source = RR / f'{tag}__{domain}.json'
                if not source.exists():
                    continue
                data = read_json(source)
                for method, block in methods:
                    if block not in data:
                        continue
                    rows.append(dict(domain=domain, C=C, method=method,
                                     task='nowcast' if block.startswith('nowcast') else 'forecast',
                                     n_eval=data.get('n_eval'), n=data.get('n'),
                                     source=str(source),
                                     **{m: data[block].get(m) for m in METRICS}))
    return [write_csv(out, 'data_efficiency.csv', rows)]


def inventory_exports(out):
    files = list(RESULTS.rglob('*'))
    file_rows, counts = [], Counter()
    for p in files:
        if not p.is_file():
            continue
        alias = p.is_symlink()
        counts[(p.suffix, 'symlink' if alias else 'physical')] += 1
        if p.suffix not in ['.json', '.npz']:
            continue
        file_rows.append(dict(relative_path=str(p.relative_to(ROOT)), extension=p.suffix,
                              bytes=p.stat().st_size, is_symlink=alias,
                              resolved=str(p.resolve()) if alias else ''))
    summary = [dict(extension=suffix, kind=kind, count=count)
               for (suffix, kind), count in sorted(counts.items())]
    artifacts = [write_csv(out, 'inventory_files.csv', file_rows),
                 write_csv(out, 'inventory_counts.csv', summary)]
    runs = []
    for source in sorted(RR.glob('*.json')):
        if '__' not in source.stem or source.stem.endswith('.fit'):
            continue
        data = read_json(source)
        tag, domain = source.stem.split('__', 1)
        row = dict(tag=tag, domain=domain, source=str(source),
                   is_symlink=source.is_symlink(), resolved=str(source.resolve()))
        for key in ['arch', 'freq', 'L', 'L_model', 'C', 'H', 'N', 'N_model', 'n',
                    'n_eval', 'seconds', 'n_views', 'anchor_mode', 'anchor_K']:
            row[key] = data.get(key)
        for task in ['forecast', 'nowcast', 'nowcast_cl', 'forecast_2s_point', 'forecast_2s_mc']:
            for metric in METRICS:
                row[f'{task}_{metric}'] = data.get(task, {}).get(metric)
        runs.append(row)
    artifacts.append(write_csv(out, 'experiment_runs.csv', runs))
    return artifacts


def load_npz(path, keys=None):

    with np.load(path, allow_pickle=True) as data:
        names = data.files if keys is None else keys
        return {key: data[key] for key in names if key in data.files}


def key_index(data):
    return {(str(s), int(t)): i for i, (s, t) in enumerate(zip(data['index_sid'], data['index_t']))}


def source_specs(domain):

    specs = []
    for name, tag in [('recast', 'v12_final'), ('light', 'abl2_light'),
                      ('zsviews', 'chronos2_zsviews_psn'), ('naive', 'chronos2_naive'),
                      ('oracle', 'chronos2_oracle')]:
        specs.append(dict(name=name, path=RR / f'{tag}__{domain}.npz', qf='Qf',
                          qn='Qn' if name in ['recast', 'light', 'zsviews'] else None))
    specs += [dict(name='cl', path=RR / f'baselines__{domain}.npz', qf=None, qn='Qn_cl'),
              dict(name='twostage', path=RR / f'baselines__{domain}.npz', qf='Qf_2s', qn=None),
              dict(name='twostage_mc', path=RR / f'baselines__{domain}.npz', qf='Qf_mc', qn=None),
              dict(name='b2f', path=FPD / f'chronos2__{domain}.npz', qf='Q_b2f', qn=None),
              dict(name='resid_adapter', path=FPD / f'chronos2__{domain}.npz', qf='Q_ra_prov', qn=None)]
    bayes_path = RR / f'bayes__{domain}.npz'
    if bayes_path.exists():
        specs += [dict(name='bayes', path=bayes_path, qf=None, qn='Qn_bayes'),
                  dict(name='bayes_twostage', path=bayes_path, qf='Qf_2s', qn=None),
                  dict(name='bayes_twostage_mc', path=bayes_path, qf='Qf_mc', qn=None)]
    return specs


def source_indexes(specs):
    out = []
    for spec in specs:
        with np.load(spec['path'], allow_pickle=True) as data:
            if 'index_sid' not in data.files:
                continue
            out.append((spec, {(str(s), int(t)): i for i, (s, t) in enumerate(zip(data['index_sid'], data['index_t']))}))
    return out


def main_sources(domain):

    sources = {}
    for spec, index in source_indexes(source_specs(domain)):
        keys = ['index_sid', 'index_t'] + [x for x in [spec['qf'], spec['qn']] if x]
        data = load_npz(spec['path'], keys)
        sources[spec['name']] = (data, index, spec['qf'], spec['qn'])
    common = set(sources['recast'][1])
    for _, (_, index, _, _) in sources.items():
        common.intersection_update(index)
    return sources, sorted(common, key=lambda key: sources['recast'][1][key])


def loss_components(q, y, scale):
    mask = np.isfinite(y) & np.isfinite(q).all(-1)
    error = y[..., None] - q
    pb = 2 * np.mean(np.maximum(LEVELS * error, (LEVELS - 1) * error), -1) / scale[:, None]
    ae = np.abs(q[..., 4] - y) / scale[:, None]
    crps_sum = np.where(mask, pb, 0.).sum(1)
    mase_sum = np.where(mask, ae, 0.).sum(1)
    count = mask.sum(1)
    crps = np.divide(crps_sum, count, out=np.full(len(count), np.nan), where=count > 0)
    mase = np.divide(mase_sum, count, out=np.full(len(count), np.nan), where=count > 0)
    return crps, crps_sum, mase, mase_sum, count


def date_string(time, t):
    if 0 <= t < len(time):
        return str(np.datetime64(time[t], 'ns')).split('T')[0]
    return ''


def loss_exports(out, load_bundle, domains=DOMAINS):
    artifacts, checks = [], []
    aggregate = read_json(RESULTS / 'agg_v12_final.json')
    agg_names = {'recast': 'v12_final', 'light': 'abl2_light', 'zsviews': 'chronos2_zsviews_psn',
                 'naive': 'plain_naive', 'oracle': 'plain_oracle'}
    for domain in domains:
        bundle = load_bundle(domain)
        indexed = source_indexes(source_specs(domain))
        rindex = indexed[0][1]
        reference_keys = set(rindex)
        indexed = [(spec, index) for spec, index in indexed if not spec['name'].startswith('bayes') or
                   len(reference_keys & set(index)) >= .95 * len(reference_keys)]
        common_keys = set(reference_keys)
        for _, index in indexed:
            common_keys.intersection_update(index)
        common = sorted(common_keys, key=lambda key: rindex[key])
        reference = load_npz(RR / f'v12_final__{domain}.npz', ['Y', 'YN', 'S', 'EV'])
        ir = np.array([rindex[key] for key in common])
        ev = reference['EV'][ir].astype(bool)
        keys = [key for key, keep in zip(common, ev) if keep]
        ir = ir[ev]
        scale = reference['S'][ir]
        cutoff = .01 * float(np.median(scale))
        base = [dict(domain=domain, sid=sid, origin=t,
                     origin_date=date_string(bundle['series'][sid]['time'], t),
                     scale=scale[i], scale_cutoff=cutoff, above_1pct_scale=bool(scale[i] >= cutoff),
                     n_common=len(common), n_eval=len(keys),
                     n_valid_forecast_cells=int(np.isfinite(reference['Y'][ir[i]]).sum()),
                     n_valid_nowcast_cells=int(np.isfinite(reference['YN'][ir[i]]).sum()))
                for i, (sid, t) in enumerate(keys)]
        fr = [dict(row) for row in base]
        nr = [dict(row) for row in base]
        for spec, index in indexed:
            method = spec['name']
            ids = np.array([index[key] for key in keys])
            for task, qkey, truth, records in [('forecast', spec['qf'], reference['Y'][ir], fr),
                                                ('nowcast', spec['qn'], reference['YN'][ir], nr)]:
                if qkey is None:
                    continue
                with np.load(spec['path'], allow_pickle=True) as handle:
                    q = handle[qkey][ids]
                losses = loss_components(q, truth, scale)
                del q
                names = ['crps', 'crps_sum', 'mase', 'mase_sum', 'count']
                for i, row in enumerate(records):
                    row.update({f'{method}_{name}': vals[i] for name, vals in zip(names, losses)})
                agg_method = agg_names.get(method, method)
                expected = aggregate[domain].get(agg_method, {}).get(task, {})
                for metric, sums in [('CRPS_s', losses[1]), ('MASE', losses[3])]:
                    observed = float(sums.sum() / losses[4].sum())
                    target = expected.get(metric)
                    checks.append(dict(domain=domain, method=method, task=task, metric=metric,
                                       recomputed=observed, stored=target,
                                       absolute_difference=abs(observed - target) if target is not None else None,
                                       n_eval=len(keys), n_cells=int(losses[4].sum())))
        for records in [fr, nr]:
            for row in records:
                for baseline in ['naive', 'cl', 'twostage', 'twostage_mc', 'b2f', 'zsviews']:
                    if f'{baseline}_crps' in row:
                        row[f'recast_minus_{baseline}_crps'] = row['recast_crps'] - row[f'{baseline}_crps']
        artifacts.append(write_csv(out, f'paired_forecast_{domain}.csv', fr))
        artifacts.append(write_csv(out, f'paired_nowcast_{domain}.csv', nr))
        print('paired complete', domain, len(keys), flush=True)
        del bundle, indexed, reference, fr, nr, base, records, losses
        gc.collect()
    for task in ['forecast', 'nowcast']:
        paths = [out / f'paired_{task}_{domain}.csv' for domain in DOMAINS if (out / f'paired_{task}_{domain}.csv').exists()]
        headers = []
        for path in paths:
            with path.open(newline='') as handle:
                headers.extend(next(csv.reader(handle)))
        fields = list(dict.fromkeys(headers))
        n_rows = 0
        with (out / f'paired_{task}_losses.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for path in paths:
                with path.open(newline='') as source:
                    for row in csv.DictReader(source):
                        writer.writerow(row)
                        n_rows += 1
        artifacts.append({'file': f'paired_{task}_losses.csv', 'rows': n_rows, 'columns': fields})
        print(f'paired_{task}_losses.csv', n_rows, 'x', len(fields), flush=True)
    artifacts.append(write_csv(out, 'score_recompute_checks.csv', checks))
    return artifacts


def failure_exports(out, load_bundle):
    origins, series, summaries = [], [], []
    for domain in ['chng_flu', 'alfred_w', 'rtdsm_q']:
        source = RESULTS / f'diag_failure_{domain}.json'
        data = read_json(source)
        bundle = load_bundle(domain)
        summaries.append(dict(domain=domain, source=str(source), **data['overall']))
        for row in data['origins']:
            rec = dict(domain=domain, source=str(source), **row)
            rec['origin_date'] = date_string(bundle['series'][row['sid']]['time'], row['t'])
            rec['crps_difference'] = row['crps_recast'] - row['crps_naive']
            origins.append(rec)
        if isinstance(data['series'], list):
            series.extend(dict(domain=domain, source=str(source), **row) for row in data['series'])
        else:
            series.extend(dict(domain=domain, sid=sid, source=str(source), **row)
                          for sid, row in data['series'].items())
    return [write_csv(out, 'failure_origins.csv', origins),
            write_csv(out, 'failure_series.csv', series),
            write_csv(out, 'failure_overall.csv', summaries)]


def illustration_exports(out, load_bundle):
    from recast.views import build_streams, age_grid
    bundle = load_bundle('kit')
    sid = 'kit_hosp7|DE|00+'
    L, t, C, N, H = bundle['L'], 700, 160, 28, 28
    series = bundle['series'][sid]
    A, R, time = series['A'], series['R'], series['time']
    denominator = float(np.nanmax(A[t - C + 1:t + 1, L]))
    triangle = []
    for u in range(t - C + 1, t + 1):
        for age in range(L + 1):
            released = bool(np.isfinite(R[u]) and R[u] <= u + age)
            visible = bool(u + age <= t and released and np.isfinite(A[u, age]))
            triangle.append(dict(domain='kit', sid=sid, origin=t, u=u,
                                 date=date_string(time, u), report_age=age, vintage=u + age,
                                 release=R[u], released_at_vintage=released, visible=visible,
                                 value=A[u, age] if visible else None,
                                 normalized_value=A[u, age] / denominator if visible else None))
    streams = build_streams(A, R, t, L, C, N, H)
    ages = age_grid(L)
    names = ['settled_visible', 'as_of'] + [f'age_{a}' for a in ages]
    traces = []
    for j, u in enumerate(range(t - C + 1, t + H + 1)):
        common = dict(domain='kit', sid=sid, origin=t, C=C, N=N, H=H, L=L,
                      u=u, date=date_string(time, u), relative_position=u - t,
                      block='history' if u <= t - N else 'nowcast' if u <= t else 'forecast')
        traces.append(dict(common, stream='settled_retrospective', report_age=L,
                           value=A[u, L], mask=bool(np.isfinite(A[u, L]))))
        for k, name in enumerate(names):
            traces.append(dict(common, stream=name, report_age=streams['age_raw'][k, j],
                               value=streams['vals'][k, j], mask=bool(streams['mask'][k, j])))
    profiles = []
    for scope, scope_series in [('national', [series]), ('domain_pooled', list(bundle['series'].values()))]:
        for age in range(L + 1):
            ratios = []
            for ss in scope_series:
                final = ss['A'][:, L]
                denom = np.where(np.abs(final) > 0, final, np.nan)
                ratio = ss['A'][:, age] / denom
                ratios.append(ratio[np.isfinite(ratio)])
            ratio = np.concatenate(ratios)
            profiles.append(dict(domain='kit', sid=sid if scope == 'national' else 'all_series',
                                 scope=scope, report_age=age, median=float(np.median(ratio)),
                                 q10=float(np.quantile(ratio, .1)), q90=float(np.quantile(ratio, .9)),
                                 n_periods=len(ratio), use='retrospective_descriptive'))
    return [write_csv(out, 'illustration_triangle.csv', triangle),
            write_csv(out, 'illustration_streams.csv', traces),
            write_csv(out, 'illustration_completion.csv', profiles)]


def example_exports(out, load_bundle):
    rows, selections = [], []
    cases = [(*case, 'median_improvement') for case in EXAMPLES]

    failures = read_json(RESULTS / 'diag_failure_chng_flu.json')['origins']
    worst = max(failures, key=lambda r: r['crps_recast'] - r['crps_naive'])
    cases.append(('chng_flu', worst['sid'], 'US-Flu, worst deterioration', 120, 'worst_deterioration'))
    for case_id, (domain, sid, title, context, rule) in enumerate(cases, 1):
        bundle = load_bundle(domain)
        series, L = bundle['series'][sid], bundle['L']
        A, R, time = series['A'], series['R'], series['time']
        indexed = source_indexes(source_specs(domain))
        rindex = indexed[0][1]
        reference_keys = set(rindex)
        common_set = set(reference_keys)
        for spec, index in indexed:
            if not spec['name'].startswith('bayes') or len(reference_keys & set(index)) >= .95 * len(reference_keys):
                common_set.intersection_update(index)
        common = sorted(common_set, key=lambda key: rindex[key])
        z = load_npz(RR / f'v12_final__{domain}.npz', ['index_sid', 'index_t', 'Qf', 'Qn_full', 'Y', 'S', 'EV'])
        naive = load_npz(RR / f'chronos2_naive__{domain}.npz', ['index_sid', 'index_t', 'Qf'])
        baseline = load_npz(RR / f'baselines__{domain}.npz', ['index_sid', 'index_t', 'Qn_cl', 'Qf_2s'])
        kz, kn, kb = key_index(z), key_index(naive), key_index(baseline)
        keys = [key for key in common if key[0] == sid and z['EV'][kz[key]]]
        iz = np.array([kz[key] for key in keys])
        inn = np.array([kn[key] for key in keys])
        recast_loss = loss_components(z['Qf'][iz], z['Y'][iz], z['S'][iz])[0]
        naive_loss = loss_components(naive['Qf'][inn], z['Y'][iz], z['S'][iz])[0]
        improvement = naive_loss - recast_loss
        order = np.argsort(improvement, kind='stable')
        if rule == 'median_improvement':
            pick = int(order[min(int(.5 * len(order)), len(order) - 1)])
        else:
            pick = next(i for i, key in enumerate(keys) if key[1] == worst['t'])
        key = keys[pick]
        t, i = key[1], kz[key]
        N, H = z['Qn_full'].shape[1], z['Qf'].shape[1]
        selections.append(dict(case_id=case_id, domain=domain, sid=sid, title=title, origin=t,
                               origin_date=date_string(time, t), selection_rule=rule,
                               candidate_n=len(keys), improvement_quantile=.5 if rule == 'median_improvement' else None,
                               rank=int(np.where(order == pick)[0][0]) + 1,
                               naive_crps=naive_loss[pick], recast_crps=recast_loss[pick],
                               improvement=improvement[pick], context=context, N_model=N, H=H, L=L))
        for u in range(max(0, t - context + 1), min(len(A), t + H + 1)):
            common_row = dict(case_id=case_id, domain=domain, sid=sid, origin=t,
                              u=u, date=date_string(time, u), relative_position=u - t,
                              block='history' if u <= t - N else 'nowcast' if u <= t else 'forecast')
            latest = A[u, min(t - u, L)] if u <= t and np.isfinite(R[u]) and R[u] <= t else None
            rows.append(dict(common_row, trace='settled', q10=None, q50=A[u, L], q90=None))
            rows.append(dict(common_row, trace='as_of', q10=None, q50=latest, q90=None))
            if t - N < u <= t:
                q = z['Qn_full'][i, u - (t - N + 1)]
                rows.append(dict(common_row, trace='RECAST_nowcast', q10=q[0], q50=q[4], q90=q[-1]))
            if t < u <= t + H:
                j = u - t - 1
                for trace, q in [('RECAST_forecast', z['Qf'][i, j]), ('naive_forecast', naive['Qf'][kn[key], j]),
                                 ('CL_pipeline_forecast', baseline['Qf_2s'][kb[key], j])]:
                    rows.append(dict(common_row, trace=trace, q10=q[0], q50=q[4], q90=q[-1]))
            ns = baseline['Qn_cl'].shape[1]
            if t - ns < u <= t:
                q = baseline['Qn_cl'][kb[key], u - (t - ns + 1)]
                rows.append(dict(common_row, trace='chain_ladder_nowcast', q10=q[0], q50=q[4], q90=q[-1]))
        del z, naive, baseline, bundle, series, indexed
        gc.collect()
    return [write_csv(out, 'example_traces.csv', rows),
            write_csv(out, 'example_selection.csv', selections)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--stage', choices=['small', 'large', 'pairs', 'cases', 'all'], default='all')
    ap.add_argument('--domains', default=','.join(DOMAINS))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    artifacts = []
    if args.stage in ['small', 'all']:
        artifacts.extend(metric_exports(args.out))
        artifacts.extend(mechanism_exports(args.out))
        artifacts.extend(dataeff_exports(args.out))
        artifacts.extend(inventory_exports(args.out))
    if args.stage in ['large', 'pairs', 'cases', 'all']:
        import sys
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, '.')
        from recast.datasets import load_bundle
        if args.stage in ['large', 'all']:
            artifacts.extend(failure_exports(args.out, load_bundle))
            artifacts.extend(illustration_exports(args.out, load_bundle))
        if args.stage in ['large', 'pairs', 'all']:
            artifacts.extend(loss_exports(args.out, load_bundle, args.domains.split(',')))
        if args.stage in ['large', 'cases', 'all']:
            artifacts.extend(example_exports(args.out, load_bundle))
    with (args.out / f'export_manifest_{args.stage}.json').open('w') as handle:
        json.dump({'artifacts': artifacts, 'plotting_backend': 'R only',
                   'script_role': 'nonvisual inspection and CSV conversion'}, handle, indent=2)


if __name__ == '__main__':
    main()
