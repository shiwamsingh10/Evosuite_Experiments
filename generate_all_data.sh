#!/usr/bin/env bash
#This script generates experimental data for any given set of inputs from D4j
# in config.py and new_mbr_get_rankings.sh

# python3 create_bug_map.py
# python3 convert_mhs2.py
# python3 utility.py
bash new_mbr_get_rankings.sh
python3 plot_for_barinel.py
# python3 run_on_d4jkfold.py 1.6 6.0
python3 reduced_spectrum.py 1.6 6.0