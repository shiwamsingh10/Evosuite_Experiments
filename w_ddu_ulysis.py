"""This script estimates ddu vs ulysis wasted effort and compares with flakiness"""
import csv
import math
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from tabulate import tabulate

test_outcome = {
    '+':'0',  #passed test case
    '-': '1'} #failed test case

bug_map=[]
print_table = []
epsilon = 1e-7
projectid = 'chart'

# ulysis_path = '/Users/shiwamsingh/Downloads/mathUlysis/evo_Math_causalitycoverage{}/bug{}/{}/fl/fault_localization'
# ddu_path = '/Users/shiwamsingh/Downloads/mathDDU/evo_Math_ddubranch{}/bug{}/{}/fl/fault_localization'
# ulysis_matrix = '/Users/shiwamsingh/Downloads/mathUlysis/evo_Math_causalitycoverage{}/bug{}/{}/fl/fault_localization_log/Math/evosuite-VCMDDU2/{}b.{}.sfl/txt/matrix.txt'
# ddu_matrix = '/Users/shiwamsingh/Downloads/mathDDU/evo_Math_ddubranch{}/bug{}/{}/fl/fault_localization_log/Math/evosuite-DDU_BRANCH/{}b.{}.sfl/txt/matrix.txt'

ulysis_path = '/Users/shiwamsingh/Downloads/chartUlysis/evo_chart_causalitycoverage{}/bug{}/{}/fl/fault_localization'
ddu_path = '/Users/shiwamsingh/Downloads/chartDDU/evo_chart_ddubranch{}/bug{}/{}/fl/fault_localization'
ulysis_matrix = '/Users/shiwamsingh/Downloads/chartUlysis/evo_chart_causalitycoverage{}/bug{}/{}/fl/fault_localization_log/Chart/evosuite-VCMDDU2/{}b.{}.sfl/txt/matrix.txt'
ddu_matrix = '/Users/shiwamsingh/Downloads/chartDDU/evo_chart_ddubranch{}/bug{}/{}/fl/fault_localization_log/Chart/evosuite-DDU_BRANCH/{}b.{}.sfl/txt/matrix.txt'


with open('{}_bug_map.csv'.format(projectid)) as bug_map_csv_file:
    for entry in csv.reader(bug_map_csv_file):
        bug_map.append(list(entry))

def get_fl_score(path):
    fl_score_file = open(path)
    reader = csv.reader(fl_score_file)
    headers = next(reader, None)
    values = next(reader, None)
    column_idx = headers.index('median_score_of_loaded_classes')
    return values[column_idx]

def check_failing_tests(fileptr):
    original_result_vector = []
    matrix = []
    while True:
        line = fileptr.readline()
        test_vector = line.split()
        if len(test_vector) == 0:
            break
        original_result_vector.append(test_outcome[test_vector[-1]])
        matrix.append(test_vector[:-1])
    original_result_vector = np.array(original_result_vector)
    if np.all(original_result_vector == '0'):
        return True, matrix, original_result_vector
    return False, matrix, original_result_vector

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
    return bug_rank, bug_score, wwe

stored_delta_W, stored_delta_effort = [], []
for run in range(5):
    for bugid, bug_column in bug_map:
        try:
            W_ulysis = float(get_fl_score(ulysis_path.format(run + 1, bugid, run + 1)))
            W_ddu = float(get_fl_score(ddu_path.format(run + 1, bugid, run + 1)))
            u_matrix = open(ulysis_matrix.format(run + 1, bugid, run + 1, bugid, bugid), 'r')
            d_matrix = open(ddu_matrix.format(run + 1, bugid, run + 1, bugid, bugid), 'r')
            failing_in_ulysis, matrix, original_result_vector = check_failing_tests(u_matrix)
            failing_in_ddu, _, _ = check_failing_tests(d_matrix)
            if (failing_in_ulysis or failing_in_ddu):
                print('No failing test for bug{} in run{}'.format(bugid, run + 1))
                continue
        except FileNotFoundError:
            print('fault_localization file not found for bug{} in run{}'.format(bugid, run+1))
            continue
        else:
            print(W_ddu, W_ulysis)
            delta_effort = ((W_ddu - W_ulysis)/W_ddu) * 100

            cov_matrix = np.array(matrix)
            ideal_result_vector = cov_matrix[:, int(bug_column) - 1]
            original_bug_rank, original_score, original_wwe = get_oichai_score_rank(cov_matrix, original_result_vector, bug_column)
            ideal_bug_rank, ideal_score, ideal_wwe = get_oichai_score_rank(cov_matrix, ideal_result_vector, bug_column)
            
            delta_W = original_wwe - ideal_wwe
            if delta_effort > -2000:
                stored_delta_W.append(delta_W)
                stored_delta_effort.append(delta_effort)
            print_table.append([run + 1, 'Bug{}'.format(bugid), bug_column, ideal_bug_rank, ideal_score, original_bug_rank, original_score, delta_W, delta_effort])

print(tabulate(print_table, headers=['Run', 'BugID', 'Bug_Column', 'Ideal Rank', 'Ideal Score(ochiai)', 'Flaky Rank', 'Flaky Score(ochiai)', 'flakiness estimate', 'ddu_vs_ulysis'], tablefmt='orgtbl'))
print(len(print_table))
plt.scatter(stored_delta_effort, stored_delta_W)
plt.xlabel('ddu_vs_ulysis %difference score')
plt.ylabel('Flakiness estimate for ulysis')
plt.title('Run on {} repository'.format(projectid))

plt.show()