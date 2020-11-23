'Estimates impact of flakiness on fault localization'
import csv
import math
import numpy as np
from scipy.stats import rankdata
from tabulate import tabulate

#/Users/shiwamsingh/Downloads/chartUlysis/evo_chart_causalitycoverage1/bug11/1/fl/fault_localization_log/Chart/evosuite-VCMDDU2/11b.11.sfl/txt
file_path = '/Users/shiwamsingh/Downloads/chartUlysis/evo_chart_causalitycoverage{}/bug{}/{}/fl/fault_localization_log/Chart/evosuite-VCMDDU2/{}b.{}.sfl/txt/matrix.txt'
# file_path = '/Users/shiwamsingh/aigym/evo_chart_cfworld1/bug{}/1/fl/fault_localization_log/Chart/evosuite-VCMDDU2/{}b.{}.sfl/txt/matrix.txt'
# file_path = '/Users/shiwamsingh/aigym/evo_chart_cfworld2/bug{}/1/fl/fault_localization_log/Chart/evosuite-VCMDDU2/{}b.{}.sfl/txt/matrix.txt'
test_outcome = {
    '+':'0',  #passed test case
    '-': '1'} #failed test case

bug_map=[]
print_table = []
epsilon = 1e-7

with open('bug_map.csv') as bug_map_csv_file:
    for entry in csv.reader(bug_map_csv_file):
        bug_map.append(list(entry))

def get_oichai_score_rank(cov_matrix, result_vector, bug_column):
    transposed_matrix = cov_matrix.transpose()
    oichai_score, recorded_scores = [], []
    for index, column in enumerate(transposed_matrix):
        phi_ep = len([i for i, cov in enumerate(column) if cov == '1' and result_vector[i] == '0'])
        phi_ef = len([i for i, cov in enumerate(column) if cov == '1' and result_vector[i] == '1'])
        phi_np = len([i for i, cov in enumerate(column) if cov == '0' and result_vector[i] == '0'])
        phi_nf = len([i for i, cov in enumerate(column) if cov == '0' and result_vector[i] == '1'])
        function_score = phi_ef / math.sqrt((phi_ef+phi_nf)*(phi_ef+phi_ep) + epsilon)
        recorded_scores.append(function_score)
    inverted_oichai_score = 1 - np.array(recorded_scores)
    function_ranks = rankdata(inverted_oichai_score)
    for index, score in enumerate(recorded_scores):
        oichai_score.append([index + 1, score, function_ranks[index]])
    oichai_score = sorted(oichai_score, reverse=True, key=lambda elem: elem[1])
    bug_rank, bug_score = [(entry[2], entry[1]) for entry in oichai_score if entry[0] == int(bug_column)][0]
    r = len([entry[1] for entry in oichai_score if entry[1] >= bug_score])
    _, m = cov_matrix.shape
    wwe = (r - 1) / (m - 1)
    print(r, m)
    print(wwe)
    return bug_rank, bug_score, wwe

for causality in range(5):
    for bug_id, bug_column in bug_map:
        bug_id, bug_column = 15, 337
        original_result_vector = []
        rank_difference = []
        try:
            file = open(file_path.format(causality + 1, bug_id, causality + 1, bug_id, bug_id), 'r')
        except FileNotFoundError:
            continue
        else:
            matrix = []
            while True:
                line = file.readline()
                test_vector = line.split()
                if len(test_vector) == 0:
                    break
                original_result_vector.append(test_outcome[test_vector[-1]])
                matrix.append(test_vector[:-1])
            cov_matrix = np.array(matrix)
            original_result_vector = np.array(original_result_vector)
            if np.all(original_result_vector == '0'):
                print('Skipped causality {} bug{} due to no failing test case'.format(causality + 1, bug_id))
                continue

            ideal_result_vector = cov_matrix[:, int(bug_column) - 1]
            
            original_bug_rank, original_score, original_wwe = get_oichai_score_rank(cov_matrix, original_result_vector, bug_column)
            ideal_bug_rank, ideal_score, ideal_wwe = get_oichai_score_rank(cov_matrix, ideal_result_vector, bug_column)
            delta_W = original_wwe - ideal_wwe
            print('Done for causality {} bugid {} column {}'.format(causality + 1, bug_id, bug_column))
            print_table.append([causality + 1, 'Bug{}'.format(bug_id), bug_column, ideal_bug_rank, ideal_score, original_bug_rank, original_score, delta_W])
            print(tabulate(print_table, headers=['Causality', 'BugID', 'Bug_Column', 'Ideal Rank', 'Ideal Score(ochiai)', 'Flaky Rank', 'Flaky Score(ochiai)', 'Delta Wasted Effort'], tablefmt='orgtbl'))
            exit(0)

# print(tabulate(print_table, headers=['Causality', 'BugID', 'Bug_Column', 'Ideal Rank', 'Ideal Score(ochiai)', 'Flaky Rank', 'Flaky Score(ochiai)', 'Delta Wasted Effort'], tablefmt='orgtbl'))

with open('flakiness_estimate.csv','w+') as csv_output:
    csvWriter = csv.writer(csv_output,delimiter=',')
    csvWriter.writerow(['Causality', 'BugID', 'Bug_Column', 'Ideal Rank', 'Ideal Score(ochiai)', 'Flaky Rank', 'Flaky Score(ochiai)', 'Delta Wasted Effort'])
    csvWriter.writerows(print_table)