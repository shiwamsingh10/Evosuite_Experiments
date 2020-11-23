'Creates mapping between the bugid and buggy column in the generated test suite matrix.txt'
import re
import csv
# buggylines_path = '/Users/shiwamsingh/aigym/MathRepo Buggylines/Math-{}.buggy.lines'
# spectra_path = '/Users/shiwamsingh/Downloads/mathDDU/evo_Math_ddubranch{}/bug{}/{}/fl/fault_localization_log/Math/evosuite-DDU_BRANCH/{}b.{}.sfl/txt/spectra.csv'

#/Users/shiwamsingh/Downloads/chartDDU/evo_chart_ddubranch1/bug1/1/fl/fault_localization_log/Chart/evosuite-DDU_BRANCH/1b.1.sfl/txt
buggylines_path = '/Users/shiwamsingh/aigym/ChartBugs/{}.buggy.lines'
# spectra_path = '/Users/shiwamsingh/Downloads/mathDDU/evo_Math_ddubranch{}/bug{}/{}/fl/fault_localization_log/Math/evosuite-DDU_BRANCH/{}b.{}.sfl/txt/spectra.csv'
spectra_path = '/Users/shiwamsingh/Downloads/chartDDU/evo_chart_ddubranch{}/bug{}/{}/fl/fault_localization_log/Chart/evosuite-DDU_BRANCH/{}b.{}.sfl/txt/spectra.csv'

bug_map = {}
component_not_found = []
bug_id_404 = []
# projectId = 'math'
projectId = 'chart'
csv_file = '{}_bug_map.csv'.format(projectId)
num_bugs = 26
for run in range(5):
    for bugid in range(num_bugs):
        bug_file = open(buggylines_path.format(bugid+1),'r')
        try:
            print(spectra_path.format(run + 1, bugid + 1, run + 1, bugid + 1, bugid + 1))
            spectra_file = open(spectra_path.format(run + 1, bugid + 1, run + 1, bugid + 1, bugid + 1), 'r')
        except FileNotFoundError:
            print('404', spectra_path.format(run + 1, bugid + 1, run + 1, bugid + 1, bugid + 1))
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
                searchObj = re.search(r'#\d+#', line, re.M|re.I)
                if searchObj is not None:
                    match = searchObj.group()
                    match = match[1:-1]
                    bug_lines.append(int(match))
            # print(bug_lines)

            while True:
                line = spectra_file.readline()
                line_count += 1
                if len(line) == 0:
                    break
                line = line.strip()
                searchObj = re.search(r'\):\d+', line, re.M|re.I)
                if searchObj is not None:
                    match = searchObj.group()
                    match = match[2:]
                    component_columns[int(match)] = line_count - 1
            # print(component_columns)
            for bug in bug_lines:
                if bug in component_columns:
                    matrix_columns = bug_map.get(bugid + 1)
                    if matrix_columns is None:
                        matrix_columns = set()
                    matrix_columns.add(component_columns[bug])
                    bug_map[bugid + 1] = matrix_columns
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