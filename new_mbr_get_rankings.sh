#!/usr/bin/env bash

export SPECTRUM_DIR=/Users/shiwamsingh/aigym/MHS
export STACCATO_DIR=/Users/shiwamsingh/aigym/Barinel/barinel/staccato
export MBR_DIR=/Users/shiwamsingh/aigym/Barinel/barinel/mbr


# project_id="Math"
# num_bugs=106
# causality=5
# bug_map=math_map_for_bash.csv

# project_id='Time'
# num_bugs=27
# causality=5
# bug_map=time_map_for_bash.csv

# project_id='Chart'
# num_bugs=26
# causality=5
# bug_map=chart_map_for_bash.csv

project_id='Mockito'
num_bugs=38
causality=5
bug_map=mockito_map_for_bash.csv

buggy_lines=`wc -l < $bug_map`
printf "causality,bugid,buggycolumn,rank\n" > barinel_output.csv
for run in $(seq 1 $causality);
do
    for idx in $(seq 1 $buggy_lines);
    do
        bugid=`awk -F, -v id=$idx 'NR==id { print $1 }' $bug_map`
        components=(`awk -F, -v id=$idx 'BEGIN{RS="\r\n"} NR==id { for (i = 2; i <= NF; ++i) print $i}' $bug_map`)
        # echo "${components[*]}"
        file_path=$SPECTRUM_DIR/$project_id/causality$run/bug$bugid
        num_tests=`wc -l < $file_path/X`
        num_components=`awk 'NR==1 { print NF }' $file_path/X`

        $STACCATO_DIR/staccato -d=1000 $num_components $file_path/X $file_path/Y > $file_path/XYhs
        $MBR_DIR/mbr -f=$file_path/XYhs --mapd1 $num_components $file_path/X $file_path/Y > temp_file 2>&1
        results_from_line=`grep -n \*\*E temp_file | cut -d : -f 1`
        awk -v line=$results_from_line 'NR>line' temp_file >> $file_path/results
        # for buggy_component in "${components[@]}"
        # do
        #     echo $file_path
        #     rank=`awk -F'[ ]' -v line=$results_from_line -v comp=$buggy_component  'NR>line && $1==comp { print NR-line }' temp_file`
        #     printf "$run,$bugid,$buggy_component,$rank\n" >> barinel_output.csv
        # done
    done
done
