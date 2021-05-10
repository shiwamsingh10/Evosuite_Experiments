import pickle
import statistics as sc
import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy.stats import rankdata
from config import *

def create_barinel_output():
    bug_map = {}
    barinel_output_csv = open('{}_barinel_output.csv'.format(d4j_repo_lower_case),'w+')
    barinel_output = csv.writer(barinel_output_csv,delimiter=',')
    barinel_output.writerow(['causality','bugid','buggycolumn','rank'])
    with open('{}_bug_map.csv'.format(d4j_repo_lower_case)) as bug_map_csv_file:
        for entry in csv.reader(bug_map_csv_file):
            bug_id, buggy_column = int(entry[0]), int(entry[1])
            if bug_id in bug_map:
                bug_map[bug_id].append(buggy_column)
            else:
                bug_map[bug_id] = [buggy_column]
                
    file_path = '/Users/shiwamsingh/aigym/MHS/{}/causality{}/bug{}/results'
    for causality in range(1, num_coverage+1):
        for bug_id in range(1, d4j_repo_bugs + 1):
            try:
                with open(file_path.format(d4j_repo, causality, bug_id)) as barinel_result:
                    print(file_path.format(d4j_repo, causality, bug_id))
                    reader = csv.reader(barinel_result, delimiter=' ')
                    found_buggy_column, score, invalidFile = [], [], False
                    for entry in reader:
                        if not entry[0].isnumeric():
                            invalidFile = True
                            break
                        found_buggy_column.append(int(entry[0]))
                        score.append(float(entry[1]))
                    
                    if not invalidFile:
                        rankings = rankdata(1-np.array(score, dtype=np.float), method='max')
                        find_bug_columns = bug_map[bug_id]
                        for buggy_col in find_bug_columns:
                            if buggy_col in found_buggy_column:
                                index = found_buggy_column.index(buggy_col)
                                rank = rankings[index]
                                barinel_output.writerow([causality,bug_id,buggy_col,rank])
                            else:
                                barinel_output.writerow([causality,bug_id,buggy_col,''])
                    
            except FileNotFoundError:
                continue



def compare_effort_barinel(c1, c2):    
    with open('cost_to_fix.pickle', 'rb') as f:
        cost_to_fix = pickle.load(f)

    with open('bug_num_components.pickle', 'rb') as f:
        bug_num_components = pickle.load(f)

    fig, axs = plt.subplots(2,3)
    

    plt.margins(0.2)
    plt.subplots_adjust(left  = 0.05,  # the left side of the subplots of the figure
right = 0.99,    # the right side of the subplots of the figure
bottom = 0.1,   # the bottom of the subplots of the figure
top = 0.9,      # the top of the subplots of the figure
wspace = 0.1,   # the amount of width reserved for blank space between subplots
hspace = 0.4)
    fig.suptitle('{} repo effort comparision c1={}, c2={}'.format(d4j_repo, c1, c2))
    
    plt_indices = [[0,0], [0,1], [0,2], [1,0], [1,1]]
    
    s_wins, b_wins, tie = 0 , 0, 0
    sig_winner, bar_winner, tied_both = 0, 0, 0
    sig_stats, bar_stats = [], []
    ## READ Barinel Results
    barinel_output = {}  
    with open('{}_barinel_output.csv'.format(d4j_repo_lower_case)) as barinel_output_csv:
        reader = csv.reader(barinel_output_csv)
        next(reader) # skip headers
        
        for entry in reader:
            key = '{}_{}_{}'.format(entry[0], entry[1], entry[2])
            if entry[3].isnumeric():
                barinel_output[key] = int(entry[3]) - 1
            else:
                barinel_output[key] = 'NP'
    all_improvement = []
    saved_labels = []
    improve_bar_labels = []
    for coverage in range(1, num_coverage + 1):
        subplt_x, subplt_y = plt_indices[coverage - 1]
        labels, bar_labels, effort_list = [], [], []
        b_effort_list, b_bar_labels = [], []
        improvement = []
        improvement_label = []

        for key,effort in cost_to_fix.items():
            b_effort = barinel_output[key]
            
            cov, bug_id, column = key.split(sep='_')
            

            if int(cov) == coverage:
                labels.append('{}_{}'.format(bug_id, column))
                
                if b_effort == 'NP' and effort != 'NP':
                    s_wins += 1
                    improvement_label.append(r"$\phi$")
                    improvement.append(0)
                elif b_effort != 'NP' and effort == 'NP':
                    b_wins += 1
                    improvement_label.append(r"$\phi$")
                    improvement.append(0)
                elif b_effort == 'NP' and effort == 'NP':
                    tie += 1
                    improvement_label.append(r"$\phi$")
                    improvement.append(0)
                else:
                    improvement.append(((b_effort - effort) / (b_effort+1) )*100)
                    improvement_label.append('{}_{}'.format(int(effort), int(b_effort)))
                    if b_effort < effort:
                        bar_winner += 1
                        bar_stats.append(effort - b_effort)
                    elif effort < b_effort:
                        sig_winner += 1
                        sig_stats.append(b_effort - effort)
                    else:
                        tied_both += 1
                    

                if b_effort == 'NP':
                    b_effort_list.append(0)
                    b_bar_labels.append(r"$\phi$")

                else:
                    b_bar_labels.append(b_effort)
                    b_effort = b_effort / bug_num_components[int(bug_id)] #Normalize
                    b_effort_list.append(b_effort)
                    
                
                
                if effort == 'NP':
                    effort_list.append(0)
                    bar_labels.append(r"$\phi$")
                else:
                    bar_labels.append(int(effort))
                    effort = effort / bug_num_components[int(bug_id)] #Normalize
                    effort_list.append(effort)
        saved_labels.append(labels)            
        all_improvement.append(improvement)
        improve_bar_labels.append(improvement_label)
        x = np.arange(len(effort_list))
        width = 0.5

        rects1 = axs[subplt_x, subplt_y].bar(x-width/2, effort_list, width, label='Sigmoid')
        rects2 = axs[subplt_x, subplt_y].bar(x+width/2, b_effort_list, width, label='Barinel')
        axs[subplt_x, subplt_y].set_xticks(x)
        axs[subplt_x, subplt_y].set_xticklabels(labels, rotation='vertical')
        # axs[subplt_x, subplt_y].set_title('C{}'.format(coverage))
        if subplt_y == 0:
            axs[subplt_x, subplt_y].set(ylabel='Effort to debug')
        
        
        for index, rect in enumerate(rects1):
            height = rect.get_height()
            axs[subplt_x, subplt_y].text(rect.get_x() + rect.get_width()/2.0, height, bar_labels[index], ha='center', va='bottom', rotation='vertical', fontsize='small')

        for index, rect in enumerate(rects2):
            height = rect.get_height()
            axs[subplt_x, subplt_y].text(rect.get_x() + rect.get_width()/2.0, height, b_bar_labels[index], ha='center', va='bottom', rotation='vertical',fontsize='small')

        axs[subplt_x, subplt_y].legend()

        
    print('Statistics')
    print('Sigmoid NON PHI cases = ', s_wins)
    print('Barinel NON PHI cases = ', b_wins)
    print('Both PHI cases = ', tie)

    sig_stats, bar_stats = np.array(sig_stats), np.array(bar_stats)
    print('Sigmoid win cases = {} mean_improvement = {} median = {}'.format(sig_winner, np.mean(sig_stats), np.median(sig_stats)))
    print('Barinel win cases = {} mean_improvement = {} median = {}'.format(bar_winner, np.mean(bar_stats), np.median(bar_stats)))
    print('Tied cases', tied_both)


    # print(improvement)
    # print('Median improvement = ', np.median( np.array(improvement)))
    # fig.tight_layout()
    plt.show()
    fig2, axs2 = plt.subplots(2,3)
    fig2.suptitle('{} repo improve comparision c1={}, c2={}'.format(d4j_repo, c1, c2))
    for coverage in range(5):
        subplt_x, subplt_y = plt_indices[coverage - 1]     
        x = np.arange(len(all_improvement[coverage]))
        rects = axs2[subplt_x, subplt_y].bar(x, all_improvement[coverage])
        axs2[subplt_x, subplt_y].set_xticks(x)
        axs2[subplt_x, subplt_y].set_xticklabels(saved_labels[coverage], rotation='vertical')
        for index, rect in enumerate(rects):
            height = rect.get_height()
            axs2[subplt_x, subplt_y].text(rect.get_x() + rect.get_width()/2.0, height, improve_bar_labels[coverage][index], ha='center', va='bottom', rotation='vertical', fontsize='small')
    fig2.tight_layout()
    plt.show()

def plot_chart(c1, c2):
    with open('refined_median_ranks.pickle', 'rb') as f:
        refined_median_ranks = pickle.load(f)

    # with open('refined_median_ranks_wwe.pickle', 'rb') as f:
    #     refined_median_ranks_wwe = pickle.load(f)

    with open('output_for_causality.pickle', 'rb') as f:
        output_for_causality = pickle.load(f)
    
    with open('bug_num_components.pickle', 'rb') as f:
        bug_num_components = pickle.load(f)

    # print(output_for_causality)
    # for k,v in refined_median_ranks.items():
    #     print(k, len(v))
    # exit()
    grid_results = csv.writer(open('grid_results.csv', 'a+'),delimiter=',')
    debug_csv = csv.writer(open('debug_out.csv', 'a+'),delimiter=',')
    # print(refined_median_ranks)
    # /Users/shiwamsingh/Documents/barinel_output.csv
    barinel_output,nop_causality_from_barinel = {},{}
    else1case = 0
    else2case = 0
    with open('{}_barinel_output.csv'.format(d4j_repo_lower_case)) as barinel_output_csv:
        reader = csv.reader(barinel_output_csv)
        next(reader) # skip headers
        
        for entry in reader:
            if entry[3].isnumeric():
                key = '{}_{}_{}'.format(entry[1], entry[2])
                if key in output_for_causality:
                    causalities = output_for_causality[key]
                    if entry[0] in causalities:
                        if key in barinel_output:
                            barinel_output[key].append(int(entry[3]))
                        else:
                            barinel_output[key] = [int(entry[3])]
                    else:
                        else1case += 1
                else:
                    else2case += 1

            else:
                key = '{}_{}'.format(entry[1], entry[2])
                if key in nop_causality_from_barinel:
                    nop_causality_from_barinel[key].append(int(entry[0]))
                else:
                    nop_causality_from_barinel[key] = [int(entry[0])]

    # print(else1case, else2case)
    debug_csv.writerow([c1, c2, else1case, else2case])
    # print(barinel_output)
    # print(nop_causality_from_barinel)
    for key, no_out_from_barinel_causalities in nop_causality_from_barinel.items():
        if key in refined_median_ranks:
            ranks_generated = refined_median_ranks[key]
            causalities = output_for_causality[key]
            filtered_list = []
            for index, causality in enumerate(causalities):  #Remove ranks from our output where barinel failed
                if int(causality) not in no_out_from_barinel_causalities:
                    filtered_list.append(ranks_generated[index])
            refined_median_ranks[key] = filtered_list

    # print(refined_median_ranks)
    refined_median_ranks_wwe, barinel_output_wwe = {}, {}
    for key, values in barinel_output.copy().items():
        if len(values) == 0:
            barinel_output.pop(key)
    
    for key, values in refined_median_ranks.copy().items():
        if len(values) == 0:
            refined_median_ranks.pop(key)

    # print('Refined', refined_median_ranks)
    # print('Barinel', barinel_output)
    # print(bug_num_components)
    for key, values in barinel_output.items():
        num_components = bug_num_components[int(key.split(sep='_')[0])] #only bugid to find num_components
        values = np.array(values)
        assert(num_components > 1)
        wwe = (values - 1 ) / (num_components - 1 )
        barinel_output_wwe[key] = sc.median(wwe)
        barinel_output[key] = sc.median(values)

    for key, values in refined_median_ranks.items():
        num_components = bug_num_components[int(key.split(sep='_')[0])] #only bugid to find num_components
        values = np.array(values)
        assert(num_components > 1)
        wwe = (values - 1 ) / (num_components - 1 )
        refined_median_ranks_wwe[key] = sc.median(wwe)
        refined_median_ranks[key] = sc.median(values)

    
    plot_labels = []
    median_improvements = []
    for key, values in refined_median_ranks.items():
        if key in barinel_output:
            # median_improvements.append(((barinel_output[key] -  refined_median_ranks[key]) / barinel_output[key]) * 100)
            median_improvements.append(((barinel_output_wwe[key] -  refined_median_ranks_wwe[key]) / barinel_output_wwe[key]) * 100)
            plot_labels.append(key)
    # print(median_improvements)
    # print(sc.mean(median_improvements))

    grid_results.writerow([c1, c2, sc.mean(median_improvements)])
    x = np.arange(len(median_improvements))
    fig, axs = plt.subplots()
    rects = axs.bar(x, median_improvements)
    for index, rect in enumerate(rects):
        key = plot_labels[index]
        height = rect.get_height()
        axs.text(rect.get_x() + rect.get_width()/2.0, height,'%d_%d' % (refined_median_ranks[key],barinel_output[key]), ha='center', va='bottom', rotation='vertical')
    axs.set(ylabel='% Improvement in Rank')
    axs.set_xticks(x)
    # axs.set_yticks(np.arange(-1000, 101, 100))
    axs.set_xticklabels(plot_labels, rotation='vertical')
    axs.set_title('{} Repo c1={} c2={}'.format(d4j_repo, c1, c2))
    plt.show()
    # plt.savefig('grid_plots/plot_chart_c1_{}_c2_{}.png'.format(int(c1*10),int(c2)),bbox_inches ="tight")
    # plt.savefig('grid_plots/plot_{}_c1_{}_c2_{}.png'.format(d4j_repo,int(c1*10),int(c2)),bbox_inches ="tight")

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# plot_chart(2.0,100)
def check_validity():
    #checking if num of components remain same across runs
    file_path='/Users/shiwamsingh/aigym/d4jKFold/ALL_SPECTRUMS/{}_bug{}_cov{}_Spectrum.txt'
    for bugid in range(d4j_repo_bugs):
        values = []
        for causality in range(1, num_coverage+1):
            file_dir=file_path.format(d4j_repo, bugid, causality)
            try:
                file = open(file_dir, 'r')
            except FileNotFoundError:
                continue
            else:
                matrix = []
                while True:
                    line = file.readline()
                    test_vector = line.split()
                    if len(test_vector) == 0:
                        break
                    matrix.append(list(map(int, test_vector[:-1])))
                
                cov_matrix = np.array(matrix)
                values.append(cov_matrix.shape[1])

        if len(values)>0:
            result = np.all(np.array(values) == values[0])
            if not result:
                print(bugid,bcolors.FAIL + ' Fail' + bcolors.ENDC, values )
            else:
                print(bugid,bcolors.OKGREEN + ' Pass' + bcolors.ENDC )



if __name__ == "__main__":
    # check_validity()
    create_barinel_output()