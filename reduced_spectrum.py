from plot_for_barinel import plot_chart
import sys
import csv
import numpy as np
import math
from scipy.stats import rankdata
import pickle
from plot_for_barinel import plot_chart, compare_effort_barinel
import matplotlib.pyplot as plt
from config import *

spectrum_path = '/Users/shiwamsingh/aigym/d4jKFold/KFold_with_reduced_Spectrum_and_ComponentMap/F{}/{}_bug{}_cov{}_Spectrum.txt'
reduced_spectrum_path = '/Users/shiwamsingh/aigym/d4jKFold/KFold_with_reduced_Spectrum_and_ComponentMap/F{}/{}_bug{}_cov{}_Reduced_Spectrum.txt'
reduced_component_map =  '/Users/shiwamsingh/aigym/d4jKFold/KFold_with_reduced_Spectrum_and_ComponentMap/F{}/{}_bug{}_cov{}_Reduced_Spectrum_ComponentMap.txt'

# file_path='/Users/shiwamsingh/aigym/d4jKFold/ALL_SPECTRUMS/{}_bug{}_cov{}_Spectrum.txt'

test_outcome = {
    '+':'0',  #passed test case
    '-': '1'} #failed test case

bug_map = {}
with open('{}_bug_map.csv'.format(d4j_repo_lower_case)) as bug_map_csv_file:
    for entry in csv.reader(bug_map_csv_file):
        bug_id, buggy_column = int(entry[0]), int(entry[1])
        if bug_id in bug_map:
            bug_map[bug_id].append(buggy_column)
        else:
            bug_map[bug_id] = [buggy_column]


c1 = float(sys.argv[1])
c2 = float(sys.argv[2])


def sigmoid(x):
    return 1/(1+np.exp(-c1*(x - c2)))

def vanilla_ochiai_ranking(cov_matrix, result_vector):
    transposed_matrix = cov_matrix.transpose()
    recorded_scores = []
    for column in transposed_matrix:
        phi_ep = len([i for i, cov in enumerate(column) if cov == 1 and result_vector[i] == '0'])
        phi_ef = len([i for i, cov in enumerate(column) if cov == 1 and result_vector[i] == '1'])
        # phi_np = len([i for i, cov in enumerate(column) if cov == 0 and result_vector[i] == '0'])
        phi_nf = len([i for i, cov in enumerate(column) if cov == 0 and result_vector[i] == '1'])
        function_score = 0
        if phi_ef != 0:
            function_score = phi_ef / math.sqrt((phi_ef+phi_nf)*(phi_ef+phi_ep))
        recorded_scores.append(function_score)
    assert(cov_matrix.shape[1] == len(recorded_scores))
    inverted_ochiai_score = 1 - np.array(recorded_scores)
    function_ranks = rankdata(inverted_ochiai_score, method='max')
    return function_ranks

def probablistic_ochiai_ranking(A, R, E_original):
    S = sigmoid(R)
    
    B = np.multiply(A, S)
    Q = np.nan_to_num((np.logical_not(A)*1) * np.inf)

    E_prime = np.min(B + Q, axis=1) * E_original


    A_transpose = A.transpose()
    recorded_scores = []
    for component in A_transpose:
        component_complement = np.invert(np.array(component, dtype=bool)) * 1
        phi_ep = component.dot(1 - E_prime)
        phi_ef = component.dot(E_prime)
        # phi_np = component_complement.dot(1 - E_prime)
        phi_nf = component_complement.dot(E_prime)
        function_score = 0
        if phi_ef != 0:
            function_score = phi_ef / math.sqrt((phi_ef+phi_nf)*(phi_ef+phi_ep))
        recorded_scores.append(function_score)
    inverted_ochiai_score = 1 - np.array(recorded_scores)
    function_ranks = rankdata(inverted_ochiai_score, method='max')
    return function_ranks, E_prime

track_ranks = {}
def populate_track_ranks(coverage, bug_id, rankings):
    buggy_columns = bug_map[bug_id]
    for buggy_column in buggy_columns:
        key = '{}_{}_{}'.format(coverage, bug_id, buggy_column)
        # print(key)
        if key in track_ranks:
            track_ranks[key].append(rankings[buggy_column - 1])
        else:
            track_ranks[key] = [rankings[buggy_column - 1]]

def read_component_map(bug_id, folder, coverage):
    component_map_dir = reduced_component_map.format(folder, d4j_repo, bug_id, coverage)
    component_map = {}
    try:
        component_map_file = open(component_map_dir, 'r')
    except FileNotFoundError:
        return None
    else:
        for entry in csv.reader(component_map_file):
            reduced_column = int(entry[0])
            component_map[reduced_column] = []
            for unpacked_column in entry[1:]:
                # unpacked_rankings[int(unpacked_column)] = reduced_rankings[reduced_column]
                component_map[reduced_column].append(int(unpacked_column))
    return component_map

def populate_unpacked_rankings(num_components, component_map, reduced_rankings):
    unpacked_rankings = np.zeros(num_components)
    reduced_rankings_map = {}
    for reduced_component, reduced_component_rank in enumerate(reduced_rankings):
        reduced_rankings_map[reduced_component] = reduced_component_rank
    
    for reduced_component, original_components in component_map.items():
        ranksum = 0
        for other_reduced_component, others_rank in reduced_rankings_map.items():
            if others_rank < reduced_rankings_map[reduced_component]:
                ranksum += len(component_map[other_reduced_component])
        for original_component in original_components:
            unpacked_rankings[original_component] = ranksum + len(original_components) - 1
    return unpacked_rankings

    
bug_num_components = {}
debug_reduced_ranks_csv = csv.writer(open('{}_debug_reduced_ranks.csv'.format(d4j_repo_lower_case),'w+'),delimiter=',')
debug_original_ranks_csv = csv.writer(open('{}_debug_original_ranks.csv'.format(d4j_repo_lower_case),'w+'),delimiter=',')
cost_to_fix = {}
for bug_id in range(1,d4j_repo_bugs+1):
    for coverage in range(1, num_coverage + 1):
        for folder in range(5): #F0...F4
            reduced_result_vector, original_result_vector = [], []
            # file_dir=file_path.format(d4j_repo, bug_id, coverage) MOD
            file_dir = reduced_spectrum_path.format(folder, d4j_repo, bug_id, coverage)
            original_spectrum_dir = spectrum_path.format(folder, d4j_repo, bug_id, coverage)
            try:
                file = open(file_dir, 'r')
                original_spectrum_file = open(original_spectrum_dir, 'r')
            except FileNotFoundError:
                continue
            else:
                matrix, original_matrix = [],[]
                while True:
                    line = file.readline().strip()
                    test_vector = line.split(sep=',')
                    if len(line) == 0:
                        break
                    reduced_result_vector.append(test_vector[-1])
                    matrix.append(list(map(int, test_vector[:-1])))
                while True:
                    line = original_spectrum_file.readline().strip()
                    test_vector = line.split()
                    if len(test_vector) == 0:
                        break
                    original_matrix.append(list(map(int, test_vector[:-1])))
                    original_result_vector.append(test_outcome[test_vector[-1]])
                cov_matrix = np.array(matrix)
                original_cov_matrix = np.array(original_matrix)
                original_result_vector = np.array(list(map(int, original_result_vector)))
                bug_num_components[bug_id] = original_cov_matrix.shape[1]
                reduced_result_vector = np.array(reduced_result_vector)
                no_sigmoid_rankings = vanilla_ochiai_ranking(cov_matrix, reduced_result_vector)
                # print('{}_bug{}_cov{}_Spectrum'.format(d4j_repo, bug_id, coverage))
                component_map = read_component_map(bug_id, folder, coverage)
                unpacked_rankings = populate_unpacked_rankings(original_cov_matrix.shape[1],component_map, no_sigmoid_rankings)

                unfixed_columns = bug_map[bug_id].copy()
                min_effort,fixed_column = math.inf, -1
                for column in unfixed_columns:
                    effort_for_column = unpacked_rankings[column - 1]
                    if effort_for_column < min_effort:
                        min_effort = effort_for_column
                        fixed_column = column

                    n_failing_for_executed = np.sum((original_cov_matrix[:, column-1] == 1)*(original_result_vector == 1))
                    if n_failing_for_executed == 0:
                        cost_to_fix['{}_{}_{}'.format(coverage,bug_id, column)] = 'NP'
                unfixed_columns.remove(fixed_column)
                cost_to_fix['{}_{}_{}'.format(coverage,bug_id, fixed_column)] = min_effort
                
                
                if bug_id == 22 and coverage == 2:
                    # print('Initial Rankings')
                    # print(no_sigmoid_rankings)                
                    debug_original_ranks_csv.writerow(['Intial Round'])
                    debug_original_ranks_csv.writerow(unpacked_rankings)
                    debug_reduced_ranks_csv.writerow(['Intial Round'])
                    debug_reduced_ranks_csv.writerow(no_sigmoid_rankings)

                populate_track_ranks(coverage, bug_id, unpacked_rankings)

                sigmoid_rankings = no_sigmoid_rankings.copy()
                error_vector = np.array(list(map(int, reduced_result_vector)))
                error_vector_original = error_vector.copy()
            
                num_refinements = 0
                while(np.any(error_vector > THRESHOLD) and num_refinements < MAX_REFINEMENTS):
                    sigmoid_rankings, error_vector = probablistic_ochiai_ranking(cov_matrix, sigmoid_rankings, error_vector_original)
                    unpacked_rankings = populate_unpacked_rankings(original_cov_matrix.shape[1], component_map, sigmoid_rankings)

                    min_effort,fixed_column = math.inf, -1
                    for column in unfixed_columns:
                        effort_for_column = unpacked_rankings[column - 1]
                        if effort_for_column < min_effort:
                            min_effort = effort_for_column
                            fixed_column = column
                    if len(unfixed_columns) > 0:
                        unfixed_columns.remove(fixed_column)
                        cost_to_fix['{}_{}_{}'.format(coverage,bug_id, fixed_column)] = min_effort
                    
                    if bug_id == 22 and coverage == 2:
                        
                        # print('Sigmoid rankings round', num_refinements)
                        # print(sigmoid_rankings)
                        # debug_ranks_csv.writerow(['Sigmoid Round #{}'.format(num_refinements + 1)])
                        # debug_ranks_csv.writerow(unpacked_rankings)
                        debug_original_ranks_csv.writerow(['Sigmoid Round #{}'.format(num_refinements + 1)])
                        debug_original_ranks_csv.writerow(unpacked_rankings)
                        debug_reduced_ranks_csv.writerow(['Sigmoid Round #{}'.format(num_refinements + 1)])
                        debug_reduced_ranks_csv.writerow(sigmoid_rankings)
                    populate_track_ranks(coverage, bug_id, unpacked_rankings)
                    num_refinements += 1
                print('Done {}_bug{}_cov{}_Spectrum'.format(d4j_repo, bug_id, coverage))

max_rounds = max(map(len, track_ranks.values()))


def plot_for_effort(cost_to_fix):
    fig, axs = plt.subplots(2,3)
    plt_indices = [[0,0], [0,1], [0,2], [1,0], [1,1]]
    fig.suptitle('{} repo effort comparision'.format(d4j_repo))
    for coverage in range(1, num_coverage + 1):
        subplt_x, subplt_y = plt_indices[coverage - 1]
        labels, bar_labels, effort_list = [], [], []
        for key,effort in cost_to_fix.items():
            cov, bug_id, column = key.split(sep='_')
            if int(cov) == coverage:
                labels.append('{}_{}'.format(bug_id, column))
                if effort == 'NP':
                    effort_list.append(-1)
                    bar_labels.append('NP')
                else:
                    effort_list.append(effort)
                    bar_labels.append(effort)
        x = np.arange(len(effort_list))
        rects1 = axs[subplt_x, subplt_y].bar(x, effort_list)
        axs[subplt_x, subplt_y].set_xticks(x)
        axs[subplt_x, subplt_y].set_xticklabels(labels, rotation='vertical')
        for index, rect in enumerate(rects1):
            height = rect.get_height()
            axs[subplt_x, subplt_y].text(rect.get_x() + rect.get_width()/2.0, height, bar_labels[index], ha='center', va='bottom', rotation='vertical')
    # fig.tight_layout()
    plt.show()
def map_to_print_row(e):
    k, v = e
    print_row = []
    print_row.extend(k.split('_'))
    print_row.extend(v)
    return print_row

best_median_ranks = {}
output_for_causality = {} # keep track of which causality generated ranks for bugid_column


def map_to_coverage():
    for k,v in track_ranks.items():
        key = '{}_{}'.format(*k.split('_')[1:]) #bugid_buggyColumn
        if key in output_for_causality:
            output_for_causality[key].append(k.split('_')[0])
        else:
            output_for_causality[key] = [k.split('_')[0]]

        if key in best_median_ranks:
            best_median_ranks[key].append(min(v))
        else:
            best_median_ranks[key] = [min(v)]

print_rows = list(map(map_to_print_row, track_ranks.items()))
map_to_coverage()
# print(output_for_causality)
with open('refined_median_ranks.pickle', 'wb') as f:
    pickle.dump(best_median_ranks, f)

with open('cost_to_fix.pickle', 'wb') as f:
    pickle.dump(cost_to_fix, f)


with open('output_for_causality.pickle', 'wb') as f:
    pickle.dump(output_for_causality, f)

with open('bug_num_components.pickle', 'wb') as f:
    pickle.dump(bug_num_components, f)

with open('{}_refined_ranks.csv'.format(d4j_repo_lower_case),'w+') as csv_output:
    csvWriter = csv.writer(csv_output,delimiter=',')
    headers = ['Causality', 'BugID', 'Bug_Column']
    for i in range(max_rounds):
        headers.append('Rank#{}'.format(i + 1))
    csvWriter.writerow(headers)
    csvWriter.writerows(print_rows)

# plot_for_effort(cost_to_fix)
compare_effort_barinel(c1, c2)
# plot_chart(c1, c2)