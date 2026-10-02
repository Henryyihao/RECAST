#!/bin/bash
source /etc/network_turbo >/dev/null 2>&1
R=data/raw
mkdir -p $R/rtdsm $R/respinow $R/kit $R/eia930
exec > $R/dl_server.log 2>&1
echo "start $(date)"
if [ -s /tmp/kit_test.csv ]; then cp /tmp/kit_test.csv $R/kit/COVID-19_hospitalizations_preprocessed.csv; fi
curl -s --max-time 300 -o $R/kit/COVID-19_hospitalizations.csv "https://raw.githubusercontent.com/KITmetricslab/hospitalization-nowcast-hub/main/data-truth/COVID-19/COVID-19_hospitalizations.csv"
ls -la $R/kit
for p in icosari/sari survstat/influenza survstat/rsv agi/are survstat/pneumococcal nrz/influenza nrz/rsv cvn/influenza cvn/rsv; do
  n=$(echo $p | tr '/' '-')
  curl -s --max-time 120 -o $R/respinow/reporting_triangle-$n.csv -w "$p %{http_code} %{size_download}\n" "https://raw.githubusercontent.com/KITmetricslab/RESPINOW-Hub/main/data/$p/reporting_triangle-$n.csv"
  curl -s --max-time 120 -o $R/respinow/target-$n.csv -w "$p target %{http_code} %{size_download}\n" "https://raw.githubusercontent.com/KITmetricslab/RESPINOW-Hub/main/data/$p/target-$n.csv"
done
curl -s --max-time 60 -o $R/respinow/listing_icosari_sari.json "https://api.github.com/repos/KITmetricslab/RESPINOW-Hub/contents/data/icosari/sari"
unset http_proxy https_proxy
for v in employ ruc ipt ipm cut cum hstarts pcpi pcpix rcon rcong rcond rcons m1 m2 h wsd oli propi renti div pinti tranr sscontrib npi ptax ndpi; do
  f=$R/rtdsm/${v}mvmd.xlsx
  ct=$(curl -s -L --max-time 120 -o $f -w "%{content_type}" "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/real-time-data/data-files/xlsx/${v}mvmd.xlsx")
  case "$ct" in *spreadsheet*) echo "rtdsm M $v ok $(stat -c %s $f)";; *) echo "rtdsm M $v MISSING ($ct)"; rm -f $f;; esac
done
for v in routput rgdp noutput rcon rinvbf rinvresid rex rimp rg rgf rgsl oph ulc rcong rconnd rcons rconhh rconsnp p pcon pcong pconhh nconsnp; do
  f=$R/rtdsm/${v}qvqd.xlsx
  ct=$(curl -s -L --max-time 120 -o $f -w "%{content_type}" "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/real-time-data/data-files/xlsx/${v}qvqd.xlsx")
  case "$ct" in *spreadsheet*) echo "rtdsm Q $v ok $(stat -c %s $f)";; *) echo "rtdsm Q $v MISSING ($ct)"; rm -f $f;; esac
done
for y in 2019 2020 2021 2022 2023 2024 2025 2026; do for h in Jan_Jun Jul_Dec; do
  f=$R/eia930/EIA930_BALANCE_${y}_${h}.csv
  [ -s $f ] && continue
  curl -s --max-time 900 -o $f -w "eia $y $h %{http_code} %{size_download}\n" "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_${y}_${h}.csv"
done; done
echo "end $(date)"
