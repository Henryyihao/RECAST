import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recast import datasets as D

BUILDERS = {
    "dv":        lambda: D.build_delphi_v4("doctor-visits", "smoothed_adj_cli", "dv", L=75),
    "hosp_cov":  lambda: D.build_delphi_v4("hospital-admissions", "smoothed_adj_covid19_from_claims", "hosp_cov", L=75),

    "chng_flu":  lambda: D.build_delphi_v4("chng", "smoothed_adj_outpatient_flu", "chng_flu", L=75),
    "nhsn_flu":  lambda: D.build_delphi_v5("nhsn", "confirmed_admissions_flu_ew", "nhsn_flu", L=12, min_ref="2022-08-01"),
    "nhsn_cov":  lambda: D.build_delphi_v5("nhsn", "confirmed_admissions_covid_ew", "nhsn_cov", L=12, min_ref="2022-08-01"),
    "nhsn_rsv":  lambda: D.build_delphi_v5("nhsn", "confirmed_admissions_rsv_ew", "nhsn_rsv", L=12, min_ref="2022-08-01"),
    "nssp_flu":  lambda: D.build_delphi_v5("nssp", "pct_ed_visits_influenza", "nssp_flu", L=8),
    "nssp_cov":  lambda: D.build_delphi_v5("nssp", "pct_ed_visits_covid", "nssp_cov", L=8),
    "nssp_rsv":  lambda: D.build_delphi_v5("nssp", "pct_ed_visits_rsv", "nssp_rsv", L=8),
    "kit_hosp7": lambda: D.build_kit("kit_hosp7", L=80, roll=7, age_groups=("00+",), all_states=True),
    "kit_age7":  lambda: D.build_kit("kit_age7", L=80, roll=7, age_groups=("00-04", "05-14", "15-34", "35-59", "60-79", "80+"), all_states=False),
    "kit":       lambda: D.merge_bundles("kit", ["kit_hosp7", "kit_age7"], notes="KIT hospitalization nowcast hub: 17 regions (all ages) + 6 national age groups; 7-day rolling sum"),
    "kit_L40":   lambda: D.build_kit("kit_L40", L=40, roll=7, age_groups=("00+",), all_states=True),
    "kit_L60":   lambda: D.build_kit("kit_L60", L=60, roll=7, age_groups=("00+",), all_states=True),
    "dv_L40":    lambda: D.build_delphi_v4("doctor-visits", "smoothed_adj_cli", "dv_L40", L=40),
    "nssp":      lambda: D.merge_bundles("nssp", ["nssp_flu", "nssp_cov", "nssp_rsv"], notes="NSSP weekly % ED visits (flu, covid, rsv), states"),
    "nhsn":      lambda: D.merge_bundles("nhsn", ["nhsn_flu", "nhsn_cov", "nhsn_rsv"], notes="NHSN weekly hospital admissions (flu, covid, rsv), states"),
    "macro_m":   lambda: D.merge_bundles("macro_m", ["rtdsm_m", "alfred_m"], notes="US monthly macro vintages (Philadelphia Fed RTDSM + ALFRED)"),
    "respinow":  lambda: D.build_respinow("respinow", L=10),
    "rtdsm_m":   lambda: D.build_rtdsm("M", "rtdsm_m", L=3),
    "rtdsm_q":   lambda: D.build_rtdsm("Q", "rtdsm_q", L=3),
    "alfred_w":  lambda: D.build_alfred("alfred_w", ["ICSA", "CCSA"], "W", L=4),
    "alfred_m":  lambda: D.build_alfred("alfred_m", ["RSAFS", "RSXFS", "UNRATE", "PAYEMS", "INDPRO", "HOUST", "TOTALSA", "DGORDER", "CPIAUCSL", "PERMIT", "RRSFS"], "M", L=3),
    "eia930":    lambda: D.build_eia930("eia930", L=72),
}

names = sys.argv[1:] or list(BUILDERS)
for n in names:
    try:
        b = BUILDERS[n]()
        D.bundle_summary(b)
    except Exception:
        print("FAILED", n); traceback.print_exc()
