'Creates mapping between the bugid and buggy column in the generated test suite matrix.txt'
import re
import csv
from config import *

projectId = d4j_repo
num_bugs = d4j_repo_bugs
buggylines_path = '/Users/shiwamsingh/aigym/Buggylines/{}/buggy_lines/{}.buggy.lines'
candidate_path = '/Users/shiwamsingh/aigym/Buggylines/{}/buggy_lines/{}.candidates'
spectra_path = '/Users/shiwamsingh/aigym/UlysisFiles/evo_{}_causalitycoverage{}/bug{}/{}/fl/fault_localization_log/{}/evosuite-VCMDDU2/{}b.{}.sfl/txt/spectra.csv'

bug_map = {}
component_not_found = []
bug_id_404 = []


csv_file = '{}_bug_map.csv'.format(d4j_repo_lower_case)

for run in range(5):
    for bugid in range(num_bugs):
        bug_file = open(buggylines_path.format(projectId, bugid+1),'r')
        candidate_file,candidates_found = None, {}
        candidates = set()
        try:
            candidate_file = open(candidate_path.format(projectId, bugid+1), 'r')
            while True:
                line = candidate_file.readline()
                if len(line)==0:
                    break
                line=line.strip().split(',')
                classname, linenum = line[1].split('/')[-1].split('#')
                classname = classname[:-5] # remove .java in class name
                candidate_linenum = line[1].split('/')[-1].split('#')[-1]

                candidates.add('{}_{}'.format(classname, linenum))
        except FileNotFoundError:
            print('404 Candidate', candidate_path.format(projectId, bugid+1))

        try:
            print(spectra_path.format(projectId, run + 1, bugid + 1, run + 1, projectId, bugid + 1, bugid + 1))
            spectra_file = open(spectra_path.format(projectId, run + 1, bugid + 1, run + 1, projectId, bugid + 1, bugid + 1), 'r')
        except FileNotFoundError:
            print('404', spectra_path.format(projectId, run + 1, bugid + 1, run + 1, projectId, bugid + 1, bugid + 1))
            bug_id_404.append('{}_{}'.format(run+1, bugid+1))
            continue
        else:
            line_count = 0
            bug_lines = []
            component_columns = {}
            while True:
                line = bug_file.readline()
                if len(line) == 0:
                    break
                line = line.strip()
                
                searchObj = re.search(r'\w+.java#\d+#.*', line, re.M|re.I)
                if searchObj is not None:
                    match = searchObj.group()
                    codeline = match.split('#')
                    if codeline[-1] != 'FAULT_OF_OMISSION':
                        bug_lines.append('{}_{}'.format(codeline[0][:-5], codeline[1]))
                
                if len(candidates)>0:
                    bug_lines.extend(list(candidates))
                        
            while True:
                line = spectra_file.readline()
                line_count += 1
                if len(line) == 0:
                    break
                line = line.strip()
                # searchObj = re.search(r'\):\d+', line, re.M|re.I)
                searchObj = re.search(r'\$[A-z0-9]+', line, re.M|re.I)
                if searchObj is not None:
                    match = searchObj.group()
                    classname = match[1:]
                    searchObj = re.search(r'\):\d+', line, re.M|re.I)
                    # print(line)
                    Nline = searchObj.group()[2:]
                    component_columns['{}_{}'.format(classname, Nline)] = line_count - 1
            for bug in bug_lines:
                if bug in component_columns:
                    matrix_columns = bug_map.get(bugid+1)
                    if matrix_columns is None:
                        matrix_columns = set()
                    matrix_columns.add(component_columns[bug])
                    bug_map[bugid+1] = matrix_columns
                else:
                    key = '{}_{}_{}'.format(run+1, bugid + 1, bug)
                    component_not_found.append(key)
for key in sorted(bug_map):
    print(key, bug_map[key])


with open(csv_file, 'w') as bugfile:
    csvwriter = csv.writer(bugfile)
    for key, value in sorted(bug_map.items()):
        for bugcolumn in value:
            csvwriter.writerow([key, bugcolumn])

print("Not found fault localization files for {run_bugid}")
print(bug_id_404)
print("Not found components {run_bugid_buggyline}")
print(sorted(component_not_found))