import numpy as np
from pathlib import Path
from config import *

file_path = '/Users/shiwamsingh/aigym/d4jKFold/ALL_SPECTRUMS/{}_bug{}_cov{}_Spectrum.txt'

out_path_X = '/Users/shiwamsingh/aigym/MHS/{}/causality{}/bug{}/X'
out_path_Y = '/Users/shiwamsingh/aigym/MHS/{}/causality{}/bug{}/Y'
out_dir = '/Users/shiwamsingh/aigym/MHS/{}/causality{}/bug{}'


test_outcome = {
    '+':'.',  #passed test case
    '-': 'x'} #failed test case

for causality in range(num_coverage):
    for bug_id in range(d4j_repo_bugs):
        Path(out_dir.format(d4j_repo, causality + 1, bug_id+1)).mkdir(parents=True, exist_ok=True)
        try:
            file = open(file_path.format(d4j_repo,bug_id+1, causality + 1), 'r')
        except FileNotFoundError:
            print('Not found', causality + 1, 'bugid', bug_id + 1)
            continue
        else:
            matrix = []
            result_vector = []
            out_file_X = open(out_path_X.format(d4j_repo, causality + 1, bug_id + 1), 'w+')
            out_file_Y = open(out_path_Y.format(d4j_repo, causality + 1, bug_id + 1), 'w+')
            while True:
                line = file.readline()
                test_vector = line.split()
                if len(test_vector) == 0:
                    break
                # matrix.append(list(test_vector[:-1]))
                out_file_X.write(line[:-2])
                out_file_X.write('\n')
                out_file_Y.write(line[-2:])
                # out_file_Y.write('\n')

                # result_vector.append(test_outcome[test_vector[-1]])
                result_vector.append(test_vector[-1])
            out_file_X.close()
            out_file_Y.close()
            # cov_matrix = np.array(matrix)
            # result_vector = np.expand_dims(result_vector, axis=1)
            # spectrum = np.append(cov_matrix, result_vector, axis=1)
            # Path(out_dir.format(causality + 1, bug_id + 1)).mkdir(parents=True, exist_ok=True)
            # out_file = open(out_path.format(causality + 1, bug_id + 1), 'w+')
            # out_file = open(out_path_X.format(causality + 1, bug_id + 1), 'w+')
            # out_file.write('{} {}\n'.format(cov_matrix.shape[1], cov_matrix.shape[0]))
            # file.seek(0)
            # for line in spectrum:
            #     out_file.write(' '.join(line))
            #     out_file.write('\n')
            # out_file.close()