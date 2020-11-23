#!/usr/bin/env bash
#This script generates buggy lines for a given d4j projectid
# Identifier Number of bugs
# Chart	 	    26
# Closure	    133
# Lang		    65
# Math	        106
# Mockito	    38
# Time  	    27

export D4J_HOME=/Users/shiwamsingh/aigym/defects4j
export SLOC_HOME=/usr/local/bin/sloccount

project_id="Chart"
num_bugs=26
output_dir="/Users/shiwamsingh/aigym/ChartBugs"
for run in $(seq 1 $num_bugs); do bash /Users/shiwamsingh/aigym/fault-localization-data/d4j_integration/get_buggy_lines.sh $project_id $run $output_dir; done