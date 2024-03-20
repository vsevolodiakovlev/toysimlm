"""
Created on Sun Dec 24 11:38:56 2023

@author: seva
"""

import pandas as pd
import numpy as np
import random
import math
import statistics as st

def generate_random_sequence(distribution, size, *args, **kwargs):
    """
    Generate a sequence of random variables based on the specified distribution.

    Parameters:
    - distribution: a string specifying the distribution ('uniform', 'normal', or 'beta').
    - size: the number of random variables to generate.
    - *args: additional arguments for the distribution.
    - **kwargs: additional keyword arguments for the distribution.

    Returns:
    - A list containing the generated random variables.
    """

    if distribution == 'uniform':
        low, high = args
        return [random.uniform(low, high) for _ in range(size)]
    elif distribution == 'normal':
        mean, std_dev = args
        return list(np.random.normal(mean, std_dev, size))
    elif distribution == 'beta':
        alpha, beta = args
        return list(np.random.beta(alpha, beta, size))
    elif distribution == 'log':
        mean, std_dev = args
        return list(np.random.lognormal(mean, std_dev, size))
    else:
        raise ValueError("Invalid distribution. Choose 'uniform', 'normal', or 'beta'.")

def compatibility_distance(char1_w, char1_j, char1_max, char2_w, char2_j, char2_max):
    coeff = 1.4
    compatibility = np.exp(-coeff * np.sqrt(((char1_w - char1_j)/char1_max) ** 2 + ((char2_w - char2_j)/char2_max) ** 2))
    # compatibility = '{0:.10f}'.format(compatibility)
    return compatibility

n = 100
reps = 1000
compat_bonus = 0
spec = '_n100x1000_extra_rskill'
w_treatment = False
j_treatment = True

random.seed(0)
np.random.seed(0)

# STATS
stats = pd.DataFrame(columns=['iterat',
                              'uw_uj',
                              'uw_mj',
                              'mw_uj',
                              'mw_mj',
                              'match_fail',
                              'w_rejects',
                              'j_rejects',
                              'mututal_reject',
                              'av_aspir_w1',
                              'av_aspir_w2',
                              'av_aspir_j1',
                              'av_aspir_j2',
                              'av_compat_1',
                              'av_compat_fin',
                              'w_edu',
                              'w_skill',
                              'w_prev_earn',
                              'occupat_0',
                              'occupat_1',
                              'occupat_2',
                              'well_matched',
                              'rm_under',
                              'rm_well',
                              'rm_over',
                              'isa_under',
                              'isa_well',
                              'isa_over',
                              'pf_under',
                              'pf_well',
                              'pf_over',
                              'j_edu',
                              'j_skill',
                              'j_wage',
                              'w_treated',
                              'j_treated'
                              ])

rep = 0
print()
print('Simulation in progress...')
while rep < reps:

    print(str(round(rep/reps*100,1)) + '%')

    # WORKERS

    isco_sl = generate_random_sequence('normal', n, 2.7, 0.9)
    isco_sl = [round(num) for num in isco_sl]
    isco_sl = [1 if num < 1 else num for num in isco_sl]
    isco_sl = [4 if num > 4 else num for num in isco_sl]

    literacy = generate_random_sequence('normal', n, 272, 46)
    literacy = [round(num) for num in literacy]
    literacy = [0 if num < 0 else num for num in literacy]
    literacy = [500 if num > 500 else num for num in literacy]

    prev_earn = generate_random_sequence('log', n, 2.446547, .7075674)
    prev_earn = [round(num) for num in prev_earn]
    prev_earn = [0 if num < 0 else num for num in prev_earn]

    compat = [0]*n
    match = [9999]*n
    r_edu = [0]*n
    r_skill = [0]*n

    data = {'edu': isco_sl,
            'skill': literacy,
            'prev_earn': prev_earn,
            'compat': compat,
            'match': match,
            'r_edu' : r_edu,
            'r_skill' : r_skill}
    workers = pd.DataFrame(data)

    mean = np.mean(workers['prev_earn'])
    sd = np.std(workers['prev_earn'])
    workers['aspir'] = generate_random_sequence('normal', n, 0.5, 0.1)
    workers['aspir'] = workers['aspir'].clip(0, 1)
    workers['aspir_init'] = workers['aspir']

    mean = np.mean(workers['skill'])
    sd = np.std(workers['skill'])
    workers['occupat'] = np.random.beta((5 - ((workers['skill'] - mean) / sd) * 5).clip(0.01, 9.99),
                                        (5 + ((workers['skill'] - mean) / sd) * 5).clip(0.01, 9.99)) * 2
    workers['occupat'] = round(workers['occupat'])

    # JOBS

    isco_sl = generate_random_sequence('normal', n, 2.7, 0.9)
    isco_sl = [round(num) for num in isco_sl]
    isco_sl = [1 if num < 1 else num for num in isco_sl]
    isco_sl = [4 if num > 4 else num for num in isco_sl]

    literacy = generate_random_sequence('normal', n, 272, 46)
    literacy = [round(num) for num in literacy]
    literacy = [0 if num<0 else num for num in literacy]
    literacy = [0 if num > 500 else num for num in literacy]

    wage = generate_random_sequence('log', n, 2.446547, .7075674)
    wage = [round(num) for num in wage]
    wage = [0 if num < 0 else num for num in wage]

    compat = [0]*n
    match = [9999]*n

    data = {'edu': isco_sl,
            'skill': literacy,
            'wage': wage,
            'compat': compat,
            'match': match}
    jobs = pd.DataFrame(data)

    mean = np.mean(jobs['wage'])
    sd = np.std(jobs['wage'])
    jobs['aspir'] = generate_random_sequence('normal', n, 0.5, 0.1)
    jobs['aspir'] = jobs['aspir'].clip(0, 1)
    jobs['aspir_init'] = jobs['aspir']

    # TREATMENT

    workers['treatment'] = 0
    if w_treatment == True:
        for i in workers.index:
            if random.uniform(0, 1) <= 1 / 2:
                #workers.loc[i, 'skill'] = workers['skill'][i] + (46 * 2)
                workers.loc[i, 'edu'] = round(workers['edu'][i] + (0.9 * 2))
                workers.loc[i, 'treatment'] = 1
        workers['edu'] = workers['edu'].clip(1, 4)

    jobs['treatment'] = 0
    if j_treatment == True:
        for j in jobs.index:
            if random.uniform(0, 1) <= 1 / 2:
                jobs.loc[j, 'skill'] = jobs['skill'][j] + (46 * 2)
                #jobs.loc[j, 'edu'] = round(jobs['edu'][j] + (0.9 * 2))
                jobs.loc[j, 'treatment'] = 1
        #jobs['edu'] = jobs['edu'].clip(1, 4)

    # MATCHING
    iterat = 0
    uw_uj = 0
    uw_mj = 0
    mw_uj = 0
    mw_mj = 0
    match_fail = 0
    w_rejects = 0
    j_rejects = 0
    mututal_reject = 0
    av_aspir_w1 = workers['aspir'].mean()
    av_aspir_j1 = jobs['aspir'].mean()
    av_aspir_w2 = 0
    av_aspir_j2 = 0
    av_compat_1 = 0

    all_matched = False
    log = []
    while all_matched == False:
        iterat += 1
        if iterat == 2:
            av_compat_1 = workers['compat'].mean()
        for i in workers.index:
            if reps == 1:
                print('Unmatched workers: ' + str((round(workers['match'].value_counts()[9999], 2)/n*100) if 9999 in workers['match'].unique() else 0) + '%')

            if all_matched == False:

                # for UNMATHCED worker
                if workers['match'][i] == 9999:
                    # generate the size of network and the list network jobs
                    m = round(np.random.normal(n / 2, n / 10))
                    if m < 1:
                        m = 1
                    elif m > n-1:
                        m = n-1
                    network_index = []
                    # if there're 5% or less of empty jobs, put them in the network
                    empty_jobs = 0
                    if jobs['match'].value_counts()[9999] <= 0.1 * n:
                        for empty_job in jobs.loc[jobs['match']==9999].index:
                            network_index.append(empty_job)
                            jobs.loc[empty_job, 'aspir'] = max(0, jobs['aspir'][empty_job] - 0.05)
                            empty_jobs += 1
                    # pick jobs from the population to fill the network
                    j = 0
                    while j < m - empty_jobs:
                        k = random.randint(0, n-1)
                        if k not in network_index:
                            network_index.append(k)
                            j += 1
                    # pull the data from the jobs dataset corresponding to the network
                    network = jobs.loc[network_index]
                    # sort jobs in network by wage
                    network = network.sort_values(['match', 'wage'], ascending=False)
                    # generate a list of indicies from the jobs pool dataset
                    network_index = network.index
                    # indicator for match occuring
                    match = False
                    # indicator for unsaccessfully applying to all jobs in network
                    none_left = False
                    # loop through all jobs in network until either match occurs or no unapplied jobs left
                    k = 0
                    while (match == False) and (none_left == False):
                        # print(str(i) + ' applies for ' + str(network_index[k]) + ', number ' + str(k))
                        # get agents' characteristics and aspiration levels
                        aspir_i = workers['aspir'][i]
                        aspir_j = jobs['aspir'][network_index[k]]
                        edu_w = workers['edu'][i]
                        edu_j = jobs['edu'][network_index[k]]
                        skill_w = workers['skill'][i]
                        skill_j = jobs['skill'][network_index[k]]
                        # compute agents' compatibility
                        compat = compatibility_distance(edu_w, edu_j, 4, skill_w, skill_j, 500)
                        # for UNMATHCED job
                        if jobs['match'][network_index[k]] == 9999:
                            # compatibility is greater than aspiration for both agents
                            if (compat > aspir_i) and (compat > aspir_j):
                                # updat compatibility levels and match status
                                workers.loc[i, 'compat'] = compat + compat_bonus
                                workers.loc[i, 'match'] = network_index[k]
                                workers.loc[i, 'r_edu'] = jobs['edu'][network_index[k]]
                                workers.loc[i, 'r_skill'] = jobs['skill'][network_index[k]]
                                jobs.loc[network_index[k], 'compat'] = compat + compat_bonus
                                jobs.loc[network_index[k], 'match'] = i
                                # add a record to the log
                                log.append(str(round(compat,2)) + ': unmatched ' + str(i) + ' matched with unmatched ' + str(network_index[k]))
                                # math occurs
                                uw_uj += 1
                                match = True
                                all_matched = (9999 not in np.unique(workers['match']))
                            else:
                                # worker rejects the job
                                if (compat <= aspir_i) and (compat > aspir_j):
                                    w_rejects += 1
                                    # decrease firm's aspiration
                                    jobs.loc[network_index[k], 'aspir'] = max(0, jobs['aspir'][network_index[k]] - 0.05)
                                # firm rejects the worker
                                elif (compat > aspir_i) and (compat <= aspir_j):
                                    j_rejects += 1
                                    # decrease worker's aspiration
                                    workers.loc[i, 'aspir'] = max(0, workers['aspir'][i]-0.05)
                                # both worker and firm reject each other
                                elif (compat <= aspir_i) and (compat <= aspir_j):
                                    mututal_reject += 1
                                    # decrease aspiration of both worker and firm
                                    jobs.loc[network_index[k], 'aspir'] = max(0, jobs['aspir'][network_index[k]] - 0.05)
                                    workers.loc[i, 'aspir'] = max(0, workers['aspir'][i] - 0.05)
                                # if there's still jobs to apply, move on to the next one
                                # exit the loop otherwise
                                if k < m-1:
                                    k += 1
                                else:
                                    match_fail += 1
                                    none_left = True
                                    log.append(str(round(compat,2)) + ': unmatched ' + str(i) + ' failed to match')

                        # for MATHCED job
                        elif jobs['match'][network_index[k]] != 9999:
                            # compatibility is greater than aspiration for both agents
                            # (since job is filled, it's compatibility is used instead of aspiration)
                            if (compat > aspir_i) and (compat > jobs['compat'][network_index[k]]):
                                # fire a worker who is currently occupying the job
                                let_go = jobs['match'][network_index[k]]
                                workers.loc[let_go, 'compat'] = 0
                                workers.loc[let_go, 'match'] = 9999
                                # updat compatibility levels and match status
                                workers.loc[i, 'compat'] = compat + compat_bonus
                                workers.loc[i, 'match'] = network_index[k]
                                workers.loc[i, 'r_edu'] = jobs['edu'][network_index[k]]
                                workers.loc[i, 'r_skill'] = jobs['skill'][network_index[k]]
                                jobs.loc[network_index[k], 'compat'] = compat + compat_bonus
                                jobs.loc[network_index[k], 'match'] = i
                                # add a record to the log
                                log.append(str(round(compat, 2)) + ': unmatched ' + str(i) + ' matched with matched ' + str(network_index[k]))
                                # math occurs
                                uw_mj += 1
                                match = True
                                all_matched = (9999 not in np.unique(workers['match']))
                            else:
                                # worker rejects the job
                                if (compat <= aspir_i) and (compat > jobs['compat'][network_index[k]]):
                                    w_rejects += 1
                                    # decrease firm's aspiration
                                    jobs.loc[network_index[k], 'aspir'] = max(0, jobs['aspir'][network_index[k]] - 0.05)
                                # firm rejects the worker
                                elif (compat > aspir_i) and (compat <= jobs['compat'][network_index[k]]):
                                    j_rejects += 1
                                    # decrease worker's aspiration
                                    workers.loc[i, 'aspir'] = max(0, workers['aspir'][i] - 0.05)
                                # both worker and firm reject each other
                                elif (compat <= aspir_i) and (compat <= jobs['compat'][network_index[k]]):
                                    mututal_reject += 1
                                    # decrease aspiration of both worker and firm
                                    jobs.loc[network_index[k], 'aspir'] = max(0, jobs['aspir'][network_index[k]] - 0.05)
                                    workers.loc[i, 'aspir'] = max(0, workers['aspir'][i] - 0.05)
                                # if there's still jobs to apply, move on to the next one
                                # exit the loop otherwise
                                if k < m-1:
                                    k += 1
                                else:
                                    match_fail += 1
                                    none_left = True
                                    log.append(str(round(compat,2)) + ': unmatched ' + str(i) + ' failed to match')
                # for MATHCED worker
                elif workers['match'][i] != 9999:
                    # get worker's current job
                    current_job = workers['match'][i]
                    # generate the size of network and the list network jobs
                    m = round(np.random.normal(n / 2, n / 10))
                    if m < 1:
                        m = 1
                    elif m > n - 1:
                        m = n - 1
                    network_index = []
                    # if there're 5% or less of empty jobs, put them in the network
                    empty_jobs = 0
                    if jobs['match'].value_counts()[9999] <= 0.1 * n:
                        for empty_job in jobs.loc[jobs['match'] == 9999].index:
                            network_index.append(empty_job)
                            jobs.loc[empty_job, 'aspir'] = max(0, jobs['aspir'][empty_job] - 0.05)
                            empty_jobs += 1
                    # pick jobs from the population to fill the network
                    j = 0
                    while j < m - empty_jobs:
                        k = random.randint(0, n - 1)
                        if k not in network_index:
                            network_index.append(k)
                            j += 1
                    # pull the data from the jobs dataset corresponding to the network
                    network = jobs.loc[network_index]
                    # sort jobs in network by wage
                    network = network.sort_values(['match', 'wage'], ascending=False)
                    # generate a list of indicies from the jobs pool dataset
                    network_index = network.index
                    # indicator for match occuring
                    match = False
                    # indicator for unsaccessfully applying to all jobs in network
                    none_left = False
                    # loop through all jobs in network until either match occurs or no unapplied jobs left
                    k = 0
                    while (match == False) and (none_left == False):
                        # print(str(i) + ' applies for ' + str(network_index[k]) + ', number ' + str(k))
                        # get agents' characteristics and aspiration levels
                        aspir_i = workers['aspir'][i]
                        aspir_j = jobs['aspir'][network_index[k]]
                        edu_w = workers['edu'][i]
                        edu_j = jobs['edu'][network_index[k]]
                        skill_w = workers['skill'][i]
                        skill_j = jobs['skill'][network_index[k]]
                        # compute agents' compatibility
                        compat = compatibility_distance(edu_w, edu_j, 4, skill_w, skill_j, 500)
                        # for UNMATHCED job
                        if jobs['match'][network_index[k]] == 9999:
                            # compatibility is greater than aspiration for both agents
                            # (since worker is employed, their compatibility is used instead of aspiration)
                            if (compat > jobs['compat'][network_index[k]]) and (compat > aspir_j):
                                # quit worker's current job
                                jobs.loc[jobs['match'] == i, 'compat'] = 0
                                jobs.loc[jobs['match'] == i, 'match'] = 9999
                                # updat compatibility levels and match status
                                workers.loc[i, 'compat'] = compat + compat_bonus
                                workers.loc[i, 'match'] = network_index[k]
                                workers.loc[i, 'r_edu'] = jobs['edu'][network_index[k]]
                                workers.loc[i, 'r_skill'] = jobs['skill'][network_index[k]]
                                jobs.loc[network_index[k], 'compat'] = compat + compat_bonus
                                jobs.loc[network_index[k], 'match'] = i
                                # add a record to the log
                                log.append(str(round(compat,2)) + ': matched ' + str(i) + ' matched with unmatched ' + str(network_index[k]))
                                # math occurs
                                mw_uj += 1
                                match = True
                                all_matched = (9999 not in np.unique(workers['match']))
                            else:
                                # worker rejects the job
                                if (compat <= jobs['compat'][network_index[k]]) and (compat > aspir_j):
                                    w_rejects += 1
                                    # decrease firm's aspiration
                                    jobs.loc[network_index[k], 'aspir'] = max(0, jobs['aspir'][network_index[k]] - 0.05)
                                # firm rejects the worker
                                elif (compat > jobs['compat'][network_index[k]]) and (compat <= aspir_j):
                                    j_rejects += 1
                                    # decrease worker's aspiration
                                    workers.loc[i, 'aspir'] = max(0, workers['aspir'][i] - 0.05)
                                # both worker and firm reject each other
                                elif (compat <= jobs['compat'][network_index[k]]) and (compat <= aspir_j):
                                    mututal_reject += 1
                                    # decrease aspiration of both worker and firm
                                    jobs.loc[network_index[k], 'aspir'] = max(0, jobs['aspir'][network_index[k]] - 0.05)
                                    workers.loc[i, 'aspir'] = max(0, workers['aspir'][i] - 0.05)
                                # if there's still jobs to apply, move on to the next one
                                # exit the loop otherwise
                                if k < m-1:
                                    k += 1
                                else:
                                    match_fail += 1
                                    none_left = True
                                    log.append(str(round(compat,2)) + ': matched ' + str(i) + ' failed to match')
                        # for MATHCED job
                        elif jobs['match'][network_index[k]] != 9999:
                            # compatibility is greater than aspiration for both agents
                            # (since worker and job are matched, their current compatibilities are used instead of aspiration)
                            if (compat > jobs['compat'][network_index[k]]) and (compat > jobs['compat'][network_index[k]]):
                                # quit worker's current job
                                jobs.loc[jobs['match'] == i, 'compat'] = 0
                                jobs.loc[jobs['match'] == i, 'match'] = 9999
                                # fire a worker who is currently occupying the job
                                let_go = jobs['match'][network_index[k]]
                                workers.loc[let_go, 'compat'] = 0
                                workers.loc[let_go, 'match'] = 9999
                                # updat compatibility levels and match status
                                workers.loc[i, 'compat'] = compat + compat_bonus
                                workers.loc[i, 'match'] = network_index[k]
                                workers.loc[i, 'r_edu'] = jobs['edu'][network_index[k]]
                                workers.loc[i, 'r_skill'] = jobs['skill'][network_index[k]]
                                jobs.loc[network_index[k], 'compat'] = compat + compat_bonus
                                jobs.loc[network_index[k], 'match'] = i
                                # add a record to the log
                                log.append(str(round(compat,2)) + ': matched ' + str(i) + ' matched with matched ' + str(network_index[k]))
                                # math occurs
                                mw_mj += 1
                                match = True
                                all_matched = (9999 not in np.unique(workers['match']))
                            else:
                                # worker rejects the job
                                if (compat <= jobs['compat'][network_index[k]]) and (compat > jobs['compat'][network_index[k]]):
                                    w_rejects += 1
                                    # decrease firm's aspiration
                                    jobs.loc[network_index[k], 'aspir'] = max(0, jobs['aspir'][network_index[k]] - 0.05)
                                # firm rejects the worker
                                elif (compat > jobs['compat'][network_index[k]]) and (compat <= jobs['compat'][network_index[k]]):
                                    j_rejects += 1
                                    # decrease worker's aspiration
                                    workers.loc[i, 'aspir'] = max(0, workers['aspir'][i] - 0.05)
                                # both worker and firm reject each other
                                elif (compat <= jobs['compat'][network_index[k]]) and (compat <= jobs['compat'][network_index[k]]):
                                    mututal_reject += 1
                                    # decrease aspiration of both worker and firm
                                    jobs.loc[network_index[k], 'aspir'] = max(0, jobs['aspir'][network_index[k]] - 0.05)
                                    workers.loc[i, 'aspir'] = max(0, workers['aspir'][i] - 0.05)
                                # if there's still jobs to apply, move on to the next one
                                # exit the loop otherwise
                                if k < m-1:
                                    k += 1
                                else:
                                    match_fail += 1
                                    none_left = True
                                    log.append(str(round(compat,2)) + ': matched ' + str(i) + ' failed to match')
        """
        if iterat > 0:
            log_df = {'record': log}
            log_df = pd.DataFrame(log)
            log_df.to_csv('log' + spec +'.csv')
            workers.to_csv('workers' + spec +'.csv')
            jobs.to_csv('jobs' + spec +'.csv')
        """

    """
    mean = np.mean(workers['compat'])
    sd = np.std(workers['compat'])
    workers['well_matched'] = np.random.beta(5 - (workers['compat'] - mean) / sd / 2,
                                             5 + (workers['compat'] - mean) / sd / 2)
    workers['well_matched'] = round(workers['well_matched'])
    """

    conditions = [workers['compat'] < 0.7229727,
                  workers['compat'] >= 0.7229727]
    values = [0,
              1]
    workers['well_matched'] = np.select(conditions, values, default=math.nan)

    # INDIRECT SELF ASSESSMENT

    conditions = [workers['edu'] > workers['r_edu'],
                workers['edu'] == workers['r_edu'],
                workers['edu'] < workers['r_edu']]
    values = [1,
              0,
              -1]
    workers['isa'] = np.select(conditions, values, default=math.nan)

    # REALISED MATCHES

    SDs = 1

    conditions = []
    values = []
    for group in workers['occupat'].unique():
        mode_edu = st.mode(workers.loc[workers['occupat'] == group, 'edu'])
        conditions.append((workers['occupat'] == group))
        values.append(mode_edu)
    workers['mode_edu'] = np.select(conditions, values, default=math.nan)

    conditions = []
    values = []
    for group in workers['occupat'].unique():
        std_edu = workers.loc[workers['occupat'] == group, 'edu'].std()
        conditions.append((workers['occupat'] == group))
        values.append(std_edu)
    workers['sd_edu'] = np.select(conditions, values, default=math.nan)

    # creating variable for country-spec mean-based RM mismatch
    conditions = [
        workers['edu'] >= workers['mode_edu'] + SDs * workers['sd_edu'],
        ((workers['edu'] < workers['mode_edu'] + SDs * workers['sd_edu']) &
         (workers['edu'] >= workers['mode_edu'] - SDs * workers['sd_edu'])),
        workers['edu'] < workers['mode_edu'] - SDs * workers['sd_edu']]
    values = [
        1,
        0,
        -1]
    workers['rm'] = np.select(conditions, values, default=math.nan)

    # --------------------- REALISED MATCHES end

    # PELLIZZARI-FICHEN

    conditions = []
    values = []

    for group in workers['occupat'].unique():
        sm_min = workers.loc[
            (workers['occupat'] == group) * (workers['well_matched'] == 1) == 1, 'skill'].quantile(0.05)
        conditions.append((workers['occupat'] == group))
        values.append(sm_min)
    workers['pf_min'] = np.select(conditions, values, default=math.nan)

    conditions = []
    values = []
    for group in workers['occupat'].unique():
        sm_max = workers.loc[
            (workers['occupat'] == group) * (workers['well_matched'] == 1) == 1, 'skill'].quantile(0.95)
        conditions.append((workers['occupat'] == group))
        values.append(sm_max)
    workers['pf_max'] = np.select(conditions, values, default=math.nan)

    # creating variable for skill mismatch
    conditions = [
        (workers['skill'] < workers['pf_min']),
        ((workers['skill'] < workers['pf_max']) & (workers['skill'] >= workers['pf_min'])),
        (workers['skill'] >= workers['pf_max'])]
    values = [
        -1,
        0,
        1]
    workers['pf'] = np.select(conditions, values, default=math.nan)

    # --------------------- PELLIZZARI-FICHEN end

    av_aspir_w2 = workers['aspir'].mean()
    av_aspir_j2 = jobs['aspir'].mean()
    av_compat_fin = workers['compat'].mean()

    """
    # print info
    print()
    print('---------------------------------------')
    print('WORKERS')
    print('---------------------------------------')
    print(workers)
    print('---------------------------------------')

    print()
    print('---------------------------------------')
    print('JOBS')
    print('---------------------------------------')
    print(jobs)
    print('---------------------------------------')

    print()
    print('---------------------------------------')
    print('LOG')
    print('---------------------------------------')
    for record in log:
        print(record)
    print('---------------------------------------')

    print()
    print('---------------------------------------')
    print('SUMMARY')
    print('---------------------------------------')
    print('iterations: ' + str(iterat))
    print('UW-UJ matches: ' + str(uw_uj))
    print('UW-MJ matches: ' + str(uw_mj))
    print('MW-UJ matches: ' + str(mw_uj))
    print('MW-MJ matches: ' + str(uw_uj))
    print('Failed to match: ' + str(match_fail))
    print('Worker rejects: ' + str(w_rejects))
    print('Firm rejects: ' + str(j_rejects))
    print('Mutual rejects: ' + str(mututal_reject))
    print('Average aspir (worker, before): ' + str(av_aspir_w1))
    print('Average aspir (worker, after): ' + str(av_aspir_w2))
    print('Average aspir (job, before): ' + str(av_aspir_j1))
    print('Average aspir (job, after): ' + str(av_aspir_j2))
    print('Average compat (1st iter): ' + str(av_compat_1))
    print('Average compat (final): ' + str(av_compat_fin))
    print('---------------------------------------')
    
    """

    if iterat == 1:
        av_compat_1 = workers['compat'].mean()

    new_record = {'iterat' : iterat,
                  'uw_uj' : uw_uj,
                  'uw_mj' : uw_mj,
                  'mw_uj' : mw_uj,
                  'mw_mj' : mw_mj,
                  'match_fail' : match_fail,
                  'w_rejects' : w_rejects,
                  'j_rejects' : j_rejects,
                  'mututal_reject' : mututal_reject,
                  'av_aspir_w1' : av_aspir_w1,
                  'av_aspir_w2' : av_aspir_w2,
                  'av_aspir_j1' : av_aspir_j1,
                  'av_aspir_j2' : av_aspir_j2,
                  'av_compat_1' : av_compat_1,
                  'av_compat_fin' : av_compat_fin,
                  'w_edu': workers['edu'].mean(),
                  'w_skill': workers['skill'].mean(),
                  'w_prev_earn': workers['prev_earn'].median(),
                  'occupat_0': workers['occupat'].value_counts()[0],
                  'occupat_1': workers['occupat'].value_counts()[1],
                  'occupat_2': workers['occupat'].value_counts()[2],
                  'well_matched': workers['well_matched'].mean(),
                  'rm_under': workers['rm'].value_counts(normalize=True)[-1] if -1 in workers['rm'].value_counts(normalize=True) else 0,
                  'rm_well': workers['rm'].value_counts(normalize=True)[0] if 0 in workers['rm'].value_counts(normalize=True) else 0,
                  'rm_over': workers['rm'].value_counts(normalize=True)[1] if 1 in workers['rm'].value_counts(normalize=True) else 0,
                  'isa_under': workers['isa'].value_counts(normalize=True)[-1] if -1 in workers['isa'].value_counts(normalize=True) else 0,
                  'isa_well': workers['isa'].value_counts(normalize=True)[0] if 0 in workers['isa'].value_counts(normalize=True) else 0,
                  'isa_over': workers['isa'].value_counts(normalize=True)[1] if 1 in workers['isa'].value_counts(normalize=True) else 0,
                  'pf_under': workers['pf'].value_counts(normalize=True)[-1] if -1 in workers['pf'].value_counts(normalize=True) else 0,
                  'pf_well': workers['pf'].value_counts(normalize=True)[0] if 0 in workers['pf'].value_counts(normalize=True) else 0,
                  'pf_over': workers['pf'].value_counts(normalize=True)[1] if 1 in workers['pf'].value_counts(normalize=True) else 0,
                  'j_edu': jobs['edu'].mean(),
                  'j_skill': jobs['skill'].mean(),
                  'j_wage': jobs['wage'].median(),
                  'w_treated': 1 - workers['treatment'].value_counts(normalize=True)[0],
                  'j_treated': 1 - jobs['treatment'].value_counts(normalize=True)[0]}
    stats.loc[len(stats)] = new_record

    """
    stats.to_csv('stats' + spec +'.csv')
    """

    rep += 1

print()
print('---------------------------------------')
print('STATS')
print('---------------------------------------')
print(stats.to_string())
print('---------------------------------------')

workers.to_csv('workers' + spec +'.csv')
jobs.to_csv('jobs' + spec +'.csv')
stats.to_csv('stats' + spec +'.csv')