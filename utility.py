import csv
from config import *
bug_map=[]
other_bugs = {}
with open('{}_bug_map.csv'.format(d4j_repo_lower_case)) as bug_map_csv_file:
    for entry in csv.reader(bug_map_csv_file):
        bug_map.append(list(entry))

print_table = []
for bug_id, bug_column in bug_map:
    key = bug_id
    value = []
    for search_bug_id, search_bug_column in bug_map:
        if bug_id == search_bug_id:
            value.append(int(search_bug_column))
    other_bugs[key] = value

for key,value in other_bugs.items():
    print_table.append([key, *value])
    

with open('{}_map_for_bash.csv'.format(d4j_repo_lower_case),'w+') as csv_output:
    csvWriter = csv.writer(csv_output,delimiter=',')
    csvWriter.writerows(print_table)