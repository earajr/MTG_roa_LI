#!/bin/bash

cd /home/ncas-sat/MTG_roa_LI_pysteps/scripts

YYYY="2026"
MM="10"
DD="16"

for hh in "00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" #"00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23"
do
    for mm in "00" "10" "20" "30" "40" "50"
    do
        python ROA_MTG-specify-datetime.py $YYYY$MM$DD$hh$mm
    done
done
