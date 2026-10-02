#!/bin/bash
set -x
cd "$(dirname "$0")/.."
TAG=v12_final; CK=checkpoints/v12_main.pt
python scripts/aggregate.py --tags $TAG,abl2_light,chronos2_zsviews_psn --arch chronos2 --no_bayes --out results/agg_${TAG}.json
python scripts/make_tables.py --main results/agg_${TAG}.json --tag $TAG --name main
python scripts/aggregate.py --tags $TAG --arch chronos2 --domains kit,nhsn --bayes_subset --out results/agg_bayes.json
python scripts/make_tables_bayes.py --agg results/agg_bayes.json --tag $TAG --out results/tables/tab_bayes.tex
ABL=abl2_noviews,abl2_noage,abl2_norev,abl2_nodistill,abl2_noanchorfc,abl2_nogate2,abl2_nogate0,abl_main7500,abl2_gatefc,abl2_backfill,abl2_nobackfill,abl2_nomod,abl2_synthbase,abl2_light,abl_shift
python scripts/aggregate.py --tags abl2_main,$ABL,abl2_views4,abl2_views8 --arch chronos2 --min_scale_frac 0.01 --no_bayes --out results/agg_abl2.json
python scripts/make_tables_abl.py --agg results/agg_abl2.json --main abl2_main --tags $ABL --out results/tables/tab_ablation.tex --numbers results/tables/numbers_abl.tex
SENS=v12_anc_med,v12_anc_p9,v12_anc_mc16,v12_anc_mc16i,v12_anc_mom,v12_preonly,v12_anchoronly,v12_views2,v12_views4,v12_views8,v12_L050,v12_L075,v12_L125,v12_L150
python scripts/aggregate.py --tags $TAG,$SENS --arch chronos2 --no_bayes --out results/agg_sens.json
python scripts/make_tables_sens.py --agg results/agg_sens.json --main $TAG --prefix v12 --abl results/agg_abl2.json --out results/tables --numbers results/tables/numbers_sens.tex
for a in bolt_s bolt_b toto; do python scripts/aggregate.py --tags ${a}_v3 --arch $a --no_bayes --out results/agg_${a}.json; done
python scripts/aggregate.py --tags timemoe_v3 --arch timemoe --no_bayes --out results/agg_timemoe.json
BB=chronos2:results/agg_${TAG}.json:${TAG},bolt_s:results/agg_bolt_s.json:bolt_s_v3,bolt_b:results/agg_bolt_b.json:bolt_b_v3,toto:results/agg_toto.json:toto_v3,timemoe:results/agg_timemoe.json:timemoe_v3
python scripts/make_tables_bb.py --spec $BB
rm -f results/tables/numbers_abl.tex results/tables/numbers_sens.tex
python scripts/make_numbers.py --agg results/agg_${TAG}.json --tag $TAG --out results/tables/numbers.tex --mech results/mech/mech_v12_main.json
python scripts/numbers_extra.py --agg results/agg_sens.json --tag $TAG --timemoe results/agg_timemoe.json:timemoe_v3 --bayes results/agg_bayes.json --bb $BB --dataeff v12 --out results/tables/numbers.tex
python scripts/make_tables_failure.py --diags results/diag_failure_chng_flu.json,results/diag_failure_alfred_w.json --out results/tables/tab_failure.tex --numbers results/tables/numbers.tex
python scripts/make_tables_abl.py --agg results/agg_abl2.json --main abl2_main --tags $ABL --out results/tables/tab_ablation.tex --numbers results/tables/numbers.tex > /dev/null
python scripts/make_tables_sens.py --agg results/agg_sens.json --main $TAG --prefix v12 --abl results/agg_abl2.json --out results/tables --numbers results/tables/numbers.tex > /dev/null
python scripts/make_domains_table.py > /dev/null 2>&1
echo ASSEMBLE_DONE
