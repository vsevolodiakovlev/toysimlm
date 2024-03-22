"""
Toy simulation of labour mismatch
=================================

This script contains the main simulation function for the toy simulation of labour mismatch.

The simulation is based on the following steps:

1. Generate data for workers' and firms' characteristics.
2. Loop through all workers and firms to match them based on their characteristics.
3. Update the match status and required education and skill for the matched agents.
4. Update the aspiration levels for the agents.
5. Repeat the matching process until all agents are matched.

The simulation function returns the following outputs:

1. A DataFrame containing the simulation statistics.
2. A DataFrame containing the workers' characteristics.
3. A DataFrame containing the firms' characteristics.
4. A list containing the log of the simulation.

The simulation function takes the following parameters:

1. n : int, optional
    The number of workers and firms in the simulation. Default is 100.
2. reps : int, optional
    The number of replications of the simulation. Default is 1000.
3. seed : int, optional
    The seed for the random number generator. Default is 0.
4. spec : str, optional
    The simulation specification name. Default is '_n100x1000_notreat'.
5. w_treatment : bool, optional
    Whether to apply treatment to workers. Default is False.
6. j_treatment : bool, optional
    Whether to apply treatment to firms. Default is False.
7. w_treatment_prob : float, optional
    The probability of workers receiving treatment. Default is 0.5.
8. j_treatment_prob : float, optional
    The probability of firms receiving treatment. Default is 0.5.
9. w_skill_treatment_mag : float, optional
    The magnitude of skill treatment for workers. Default is 92.
10. w_edu_treatment_mag : float, optional
    The magnitude of education treatment for workers. Default is 1.8.
11. j_skill_treatment_mag : float, optional
    The magnitude of skill treatment for firms. Default is 92.
12. j_edu_treatment_mag : float, optional
    The magnitude of education treatment for firms. Default is 1.8.
13. well_matched_threshold : float, optional
    The threshold for well-matched agents. Default is 0.7229727.
    
The simulation function is called as follows:
    
        stats, workers, jobs, log = toymsim_lm(n = 100,
                                            reps = 1000,
                                            seed = 0,
                                            spec = '_n100x1000_notreat',
                                            w_treatment = False,
                                            j_treatment = False,
                                            w_treatment_prob = 0.5,
                                            j_treatment_prob = 0.5,
                                            w_skill_treatment_mag = 46 * 2,
                                            w_edu_treatment_mag = 0.9 * 2,
                                            j_skill_treatment_mag = 46 * 2,
                                            j_edu_treatment_mag = 0.9 * 2,
                                            well_matched_threshold=0.7229727)

The simulation function returns the following outputs:

1. stats : pandas DataFrame
    A DataFrame containing the simulation statistics.
2. workers : pandas DataFrame   
    A DataFrame containing the workers' characteristics.
3. jobs : pandas DataFrame
    A DataFrame containing the firms' characteristics.
4. log : list
    A list containing the log of the simulation.

The simulation function generates random data for workers' and firms' characteristics, such as education, skill, previous earnings, wage, compatibility, match status, required education, and required skill. It then simulates the matching process between workers and firms based on their characteristics and compatibility. The simulation tracks various statistics, such as the number of unmatched workers, matched workers, unmatched jobs, matched jobs, failed matches, rejections by workers, rejections by firms, mutual rejections, average aspiration levels, average compatibility levels, average education, average skill, average previous earnings, occupation distribution, well-matched indicator, share of under-matched workers, share of well-matched workers, share of over-matched workers, average firms' education requirement, average firms' skill requirement, average wage, share of treated workers, and share of treated firms.

The simulation function uses the following helper functions:

1. generate_random_sequence : Generate a sequence of random variables based on the specified distribution.
2. compatibility_distance : Compute the compatibility distance between two agents.

The simulation function is based on the following assumptions:

1. The simulation is based on a toy model of labour mismatch.
2. The simulation generates random data for workers' and firms' characteristics.
3. The simulation matches workers and firms based on their characteristics and compatibility.
4. The simulation tracks various statistics related to the matching process.
5. The simulation allows for the specification of treatment effects on workers and firms.
6. The simulation calculates the share of under-matched, well-matched, and over-matched workers based on different matching criteria.

The simulation function is useful for understanding the dynamics of labour mismatch and the impact of treatment effects on the matching process. The simulation can be used to explore different scenarios and test various hypotheses related to labour market outcomes.

The simulation function can be extended by adding additional features, such as different matching algorithms, more complex treatment effects, and alternative matching criteria. The simulation can also be used to analyze the sensitivity of the results to different parameter values and assumptions.

The simulation function can be applied to real-world data to analyze labour market outcomes and inform policy decisions. By simulating different scenarios and treatment effects, the simulation can help policymakers design more effective interventions to reduce labour mismatch and improve labour market efficiency.

The simulation function provides a flexible framework for studying labour market dynamics and exploring the impact of different factors on the matching process. By simulating a toy model of labour mismatch, the function allows researchers to gain insights into the complex interactions between workers and firms in the labour market.

The simulation function can be used to generate empirical predictions and test theoretical models of labour market dynamics. By comparing the simulation results to real-world data, researchers can validate the model and identify areas for further research and development.

Overall, the simulation function provides a valuable tool for studying labour market outcomes, understanding the mechanisms of labour mismatch, and exploring the effects of treatment interventions on the matching process. By simulating different scenarios and treatment effects, researchers can gain new insights into the dynamics of the labour market and inform policy decisions to improve labour market efficiency and outcomes.

Author: 
Date: 
"""

import os
import pandas as pd
import numpy as np
import random
import math
import statistics as st

def generate_random_sequence(distribution, size, *args, **kwargs):
    """
    Generate a sequence of random variables based on the specified distribution.

    Parameters
    ----------
    distribution : str
        The distribution of the random variables. Choose 'uniform', 'normal', 'beta', or 'log'.
    size : int
        The size of the sequence.
    args : list
        The parameters of the distribution.

    Returns
    -------
    list
        A list of random variables.

    Raises
    ------
    ValueError
        If the distribution is not 'uniform', 'normal', 'beta', or 'log'.

    Examples
    --------
    >>> generate_random_sequence('uniform', 10, 0, 1)
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    >>> generate_random_sequence('normal', 10, 0, 1)
    [-0.2, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.1]
    >>> generate_random_sequence('beta', 10, 2, 5)
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    >>> generate_random_sequence('log', 10, 0, 1)
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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
    """
    Compute the compatibility distance between two agents.

    Parameters
    ----------
    char1_w : float
        The characteristic of the worker.
    char1_j : float
        The characteristic of the job.
    char1_max : float
        The maximum value of the characteristic.
    char2_w : float
        The characteristic of the worker.
    char2_j : float
        The characteristic of the job.
    char2_max : float
        The maximum value of the characteristic.

    Returns
    -------
    compatibility : float
        The compatibility distance between the two agents.

    Examples
    --------
    >>> compatibility_distance(2, 3, 4, 300, 400, 500)
    """

    coeff = 1.4
    compatibility = np.exp(-coeff * np.sqrt(((char1_w - char1_j)/char1_max) ** 2 + ((char2_w - char2_j)/char2_max) ** 2))
    # compatibility = '{0:.10f}'.format(compatibility)
    return compatibility

# Simulation parameters
"""
n = 100
reps = 1000
seed = 0
spec = '_n100x1000_extra_rskill'
w_treatment = False
j_treatment = True
w_treatment_prob = 0.5
j_treatment_prob = 0.5
w_skill_treatment_mag = 46 * 2
w_edu_treatment_mag = 0.9 * 2
j_skill_treatment_mag = 46 * 2
j_edu_treatment_mag = 0.9 * 2
well_matched_threshold = 0.7229727
"""

def simulate(n = 100,
              reps = 1000,
              seed = 0,
              spec = '_n100x1000_notreat',
              w_treatment = False,
              j_treatment = False,
              w_treatment_prob = 0.5,
              j_treatment_prob = 0.5,
              w_skill_treatment_mag = 46 * 2,
              w_edu_treatment_mag = 0.9 * 2,
              j_skill_treatment_mag = 46 * 2,
              j_edu_treatment_mag = 0.9 * 2,
              well_matched_threshold=0.7229727):

    """
    Run the toy simulation of labour mismatch.

    Parameters
    ----------
    n : int, optional
        The number of workers and firms in the simulation. Default is 100.
    reps : int, optional
        The number of replications of the simulation. Default is 1000.
    seed : int, optional
        The seed for the random number generator. Default is 0.
    spec : str, optional
        The simulation specification name. Default is '_n100x1000_notreat'.
    w_treatment : bool, optional
        Whether to apply treatment to workers. Default is False.
    j_treatment : bool, optional
        Whether to apply treatment to firms. Default is False.
    w_treatment_prob : float, optional
        The probability of workers receiving treatment. Default is 0.5.
    j_treatment_prob : float, optional
        The probability of firms receiving treatment. Default is 0.5.
    w_skill_treatment_mag : float, optional
        The magnitude of skill treatment for workers. Default is 92.
    w_edu_treatment_mag : float, optional
        The magnitude of education treatment for workers. Default is 1.8.
    j_skill_treatment_mag : float, optional
        The magnitude of skill treatment for firms. Default is 92.
    j_edu_treatment_mag : float, optional
        The magnitude of education treatment for firms. Default is 1.8.
    well_matched_threshold : float, optional
        The threshold for well-matched agents. Default is 0.7229727.

    Returns
    -------
    stats : pandas DataFrame
        A DataFrame containing the simulation statistics.
    workers : pandas DataFrame
        A DataFrame containing the workers' characteristics.
    jobs : pandas DataFrame
        A DataFrame containing the firms' characteristics.
    log : list
        A list containing the log of the simulation.

    Examples
    --------
    >>> stats, workers, jobs, log = toymsim_lm(n = 100,
                                                reps = 1000,
                                                seed = 0,
                                                spec = '_n100x1000_notreat',
                                                w_treatment = False,
                                                j_treatment = False,
                                                w_treatment_prob = 0.5,
                                                j_treatment_prob = 0.5,
                                                w_skill_treatment_mag = 46 * 2,
                                                w_edu_treatment_mag = 0.9 * 2,
                                                j_skill_treatment_mag = 46 * 2,
                                                j_edu_treatment_mag = 0.9 * 2,
                                                well_matched_threshold=0.7229727)
    """

    # Seeds
    random.seed(seed)
    np.random.seed(seed)

    # Simulation stats dataframe
    stats = pd.DataFrame(columns=['iterat', # iteration counter
                                'uw_uj', # unmatched workers matching with unmatched job count
                                'uw_mj', # unmatched workers matching with matched job count
                                'mw_uj', # matched workers matching with unmatched job count
                                'mw_mj', # matched workers matching with matched job count
                                'match_fail', # failed matches count
                                'w_rejects', # rejections by the workers count
                                'j_rejects', # rejections by the firms count
                                'mututal_reject', # mutual rejections count
                                'av_aspir_w1', # average aspiration level for workers before the simulation
                                'av_aspir_w2', # average aspiration level for workers after the simulation
                                'av_aspir_j1', # average aspiration level for firms before the simulation
                                'av_aspir_j2', # average aspiration level for firms after the simulation
                                'av_compat_1', # average compatibility level for workers after the first iteration (zero in the first iteration)
                                'av_compat_fin', # average compatibility level for workers after the simulation
                                'w_edu', # average workers' education
                                'w_skill', # average workers' skill
                                'w_prev_earn', # average workers' previous earnings
                                'occupat_0', # number of workers in occupation 0
                                'occupat_1', # number of workers in occupation 1
                                'occupat_2', # number of workers in occupation 2
                                'well_matched', # well-matched indicator
                                'rm_under', # share of under-matched workers according to RM
                                'rm_well', # share of well-matched workers according to RM
                                'rm_over', # share of over-matched workers according to RM
                                'isa_under', # share of under-matched workers according to ISA
                                'isa_well', # share of well-matched workers according to ISA
                                'isa_over', # share of over-matched workers according to ISA
                                'pf_under', # share of under-matched workers according to PF
                                'pf_well', # share of well-matched workers according to PF
                                'pf_over', # share of over-matched workers according to PF
                                'j_edu', # average firms' education requirement
                                'j_skill', # average firms' skill requirement
                                'j_wage', # average wage
                                'w_treated', # share of treated workers
                                'j_treated' # share of treated firms
                                ])

    # Set rep counter to 0
    rep = 0
    print()
    print('Simulation in progress...')

    # Simulation loop
    while rep < reps:

        # Progress message
        print(str(round(rep/reps*100,1)) + '%')

        # ------------------------------------------
        # Generate data for workers' characterictics
        # ------------------------------------------

        # Education
        isco_sl = generate_random_sequence('normal', n, 2.7, 0.9)
        isco_sl = [round(num) for num in isco_sl]
        isco_sl = [1 if num < 1 else num for num in isco_sl]
        isco_sl = [4 if num > 4 else num for num in isco_sl]

        # Skill
        literacy = generate_random_sequence('normal', n, 272, 46)
        literacy = [round(num) for num in literacy]
        literacy = [0 if num < 0 else num for num in literacy]
        literacy = [500 if num > 500 else num for num in literacy]

        # Previous earnings
        prev_earn = generate_random_sequence('log', n, 2.446547, .7075674)
        prev_earn = [round(num) for num in prev_earn]
        prev_earn = [0 if num < 0 else num for num in prev_earn]

        # Compatibility
        compat = [0]*n

        # Match status
        match = [9999]*n

        # Required education and skill
        r_edu = [0]*n
        r_skill = [0]*n

        # Put the data into a dataframe
        data = {'edu': isco_sl,
                'skill': literacy,
                'prev_earn': prev_earn,
                'compat': compat,
                'match': match,
                'r_edu' : r_edu,
                'r_skill' : r_skill}
        workers = pd.DataFrame(data)

        # Make sure compatibility is a float
        workers['compat'] = workers['compat'].astype(float)

        # !!! potentially redundant !!!
        # mean = np.mean(workers['prev_earn'])
        # sd = np.std(workers['prev_earn'])

        # Aspiration (selectivity in the slides)
        workers['aspir'] = generate_random_sequence('normal', n, 0.5, 0.1)
        workers['aspir'] = workers['aspir'].clip(0, 1)

        # Initial aspiration
        workers['aspir_init'] = workers['aspir']

        # Occupation
        mean = np.mean(workers['skill'])
        sd = np.std(workers['skill'])
        workers['occupat'] = np.random.beta((5 - ((workers['skill'] - mean) / sd) * 5).clip(0.01, 9.99),
                                            (5 + ((workers['skill'] - mean) / sd) * 5).clip(0.01, 9.99)) * 2
        workers['occupat'] = round(workers['occupat'])

        # ----------------------------------------
        # Generate data for firms' characterictics
        # ----------------------------------------

        # Education requirement
        isco_sl = generate_random_sequence('normal', n, 2.7, 0.9)
        isco_sl = [round(num) for num in isco_sl]
        isco_sl = [1 if num < 1 else num for num in isco_sl]
        isco_sl = [4 if num > 4 else num for num in isco_sl]

        # Skill requirement
        literacy = generate_random_sequence('normal', n, 272, 46)
        literacy = [round(num) for num in literacy]
        literacy = [0 if num<0 else num for num in literacy]
        literacy = [0 if num > 500 else num for num in literacy]

        # Wage
        wage = generate_random_sequence('log', n, 2.446547, .7075674)
        wage = [round(num) for num in wage]
        wage = [0 if num < 0 else num for num in wage]

        # Compatibility
        compat = [0]*n

        # Match status
        match = [9999]*n

        # Put the data into a dataframe
        data = {'edu': isco_sl,
                'skill': literacy,
                'wage': wage,
                'compat': compat,
                'match': match}
        jobs = pd.DataFrame(data)

        # Make sure compatibility is a float
        jobs['compat'] = jobs['compat'].astype(float)

        # !!! potentially redundant !!!
        # mean = np.mean(jobs['wage'])
        # sd = np.std(jobs['wage'])
        
        # Aspiration (selectivity in the slides)
        jobs['aspir'] = generate_random_sequence('normal', n, 0.5, 0.1)
        jobs['aspir'] = jobs['aspir'].clip(0, 1)
        jobs['aspir_init'] = jobs['aspir']

        # -------------------------------- 
        # Treatment specification
        # --------------------------------

        # Workers' treatment
        workers['treatment'] = 0
        if w_treatment == True:
            for i in workers.index:
                if random.uniform(0, 1) <= w_treatment_prob:
                    # Skill treatment
                    workers.loc[i, 'skill'] = workers['skill'][i] + w_skill_treatment_mag
                    # Education treatment
                    workers.loc[i, 'edu'] = round(workers['edu'][i] + w_edu_treatment_mag)
                    # Treatment indicator
                    workers.loc[i, 'treatment'] = 1
            if w_edu_treatment_mag != 0:
                workers['edu'] = workers['edu'].clip(1, 4)

        jobs['treatment'] = 0
        if j_treatment == True:
            for j in jobs.index:
                if random.uniform(0, 1) <= j_treatment_prob:
                    # Skill treatment
                    jobs.loc[j, 'skill'] = jobs['skill'][j] + j_skill_treatment_mag
                    # Education treatment
                    jobs.loc[j, 'edu'] = round(jobs['edu'][j] + j_edu_treatment_mag)
                    # Treatment indicator
                    jobs.loc[j, 'treatment'] = 1
            if j_edu_treatment_mag != 0:
                jobs['edu'] = jobs['edu'].clip(1, 4)

        # Simulation stats data
        iterat = 0 # iteration counter
        uw_uj = 0 # unmatched workers matching with unmatched job count
        uw_mj = 0 # unmatched workers matching with matched job count
        mw_uj = 0 # matched workers matching with unmatched job count
        mw_mj = 0 # matched workers matching with matched job count
        match_fail = 0 # failed matches count
        w_rejects = 0 # rejections by the workers count
        j_rejects = 0 # rejections by the firms count
        mututal_reject = 0 # mutual rejections count
        av_aspir_w1 = workers['aspir'].mean() # average aspiration level for workers before the simulation
        av_aspir_j1 = jobs['aspir'].mean() # average aspiration level for firms before the simulation
        av_aspir_w2 = 0 # average aspiration level for workers after the simulation
        av_aspir_j2 = 0 # average aspiration level for firms after the simulation
        av_compat_1 = 0 # average compatibility level for workers after the first iteration (zero in the first iteration)

        # All matched indicator
        all_matched = False

        # Create log
        log = []

        # Matching loop
        while all_matched == False:
            iterat += 1
            
            # !!! potentially redundant !!!
            # Update average compatibility level for workers after the first iteration
            if iterat == 2:
                av_compat_1 = workers['compat'].mean()
            
            # Loop through all workers
            for i in workers.index:

                # For a single rep simulation, print the percentage of unmatched workers
                if reps == 1:
                    print('Unmatched workers: ' + str((round(workers['match'].value_counts()[9999], 2)/n*100) if 9999 in workers['match'].unique() else 0) + '%')

                # Check if all workers are matched
                if all_matched == False:

                    # --------------------------------
                    # For an unmatched worker
                    # --------------------------------

                    # Check if the worker is unmatched
                    if workers['match'][i] == 9999:
                        
                        # Generate the size of network
                        m = round(np.random.normal(n / 2, n / 10))
                        if m < 1:
                            m = 1
                        elif m > n-1:
                            m = n-1

                        # List for network jobs
                        network_index = []

                        # Check if the number of empty jobs is 10% or less
                        empty_jobs = 0
                        if jobs['match'].value_counts()[9999] <= 0.1 * n:
                            for empty_job in jobs.loc[jobs['match']==9999].index:
                                # Add empty jobs to the network
                                network_index.append(empty_job)
                                # Penalize the aspiration level of the empty job
                                jobs.loc[empty_job, 'aspir'] = max(0, jobs['aspir'][empty_job] - 0.05)
                                # Update the count of empty jobs
                                empty_jobs += 1

                        j = 0
                        # Check if there is space left in the network
                        while j < m - empty_jobs:
                            # Randomly pick a job from the population
                            k = random.randint(0, n-1)
                            # Check if the job is not already in the network
                            if k not in network_index:
                                # Add the job to the network
                                network_index.append(k)
                                j += 1

                        # Pull the data from the jobs dataset corresponding to the network
                        network = jobs.loc[network_index]
                        
                        # Sort jobs in network by wage
                        network = network.sort_values(['match', 'wage'], ascending=False)
                        
                        # Generate a list of indicies from the jobs pool dataset
                        network_index = network.index

                        # Indicator for a match occuring
                        match = False
                        
                        # Indicator for being rejected by all jobs in the network
                        none_left = False
                        
                        # Loop through all jobs in network until either a match occurs or no unapplied jobs left
                        k = 0
                        # Check if no match has occured and there are still jobs in the network
                        while (match == False) and (none_left == False):
                            
                            # !!! potentially redundant !!!
                            # print(str(i) + ' applies for ' + str(network_index[k]) + ', number ' + str(k))
                            
                            # Pull the agents' characteristics from the dataset
                            aspir_i = workers['aspir'][i]
                            aspir_j = jobs['aspir'][network_index[k]]
                            edu_w = workers['edu'][i]
                            edu_j = jobs['edu'][network_index[k]]
                            skill_w = workers['skill'][i]
                            skill_j = jobs['skill'][network_index[k]]
                            
                            # Compute agents' compatibility
                            compat = compatibility_distance(edu_w, edu_j, 4, skill_w, skill_j, 500)

                            # --------------------------------
                            # For an unmatched job
                            # --------------------------------
                            
                            # Check if the job is unmatched
                            if jobs['match'][network_index[k]] == 9999:
                                
                                # Check if compatibility is greater than aspiration for both agents
                                if (compat > aspir_i) and (compat > aspir_j):
                                    # Match occurs

                                    # Update compatibility levels
                                    workers.loc[i, 'compat'] = compat
                                    jobs.loc[network_index[k], 'compat'] = compat

                                    # Update match status
                                    workers.loc[i, 'match'] = network_index[k]
                                    jobs.loc[network_index[k], 'match'] = i

                                    # Update required education and skill
                                    workers.loc[i, 'r_edu'] = jobs['edu'][network_index[k]]
                                    workers.loc[i, 'r_skill'] = jobs['skill'][network_index[k]]

                                    # Add a record to the log
                                    log.append(str(round(compat,2)) + ': unmatched ' + str(i) + ' matched with unmatched ' + str(network_index[k]))
                                    
                                    # Update the count of unmatched workers matching with unmatched jobs
                                    uw_uj += 1

                                    # Update the match indicator
                                    match = True

                                    # Update the all matched indicator
                                    all_matched = (9999 not in np.unique(workers['match']))
                                else:
                                    # Match fails
                                    
                                    # Check if the worker rejects the job
                                    if (compat <= aspir_i) and (compat > aspir_j):
                                        # Update the count of rejections by the workers
                                        w_rejects += 1
                                        # Penalize aspiration of the firm
                                        jobs.loc[network_index[k], 'aspir'] = max(0, jobs['aspir'][network_index[k]] - 0.05)
                                    
                                    # Check if the firm rejects the worker
                                    elif (compat > aspir_i) and (compat <= aspir_j):
                                        # Update the count of rejections by the firms
                                        j_rejects += 1
                                        # Penalize aspiration of the worker
                                        workers.loc[i, 'aspir'] = max(0, workers['aspir'][i]-0.05)

                                    # Check if both worker and firm reject each other
                                    elif (compat <= aspir_i) and (compat <= aspir_j):
                                        # Update the count of mutual rejections
                                        mututal_reject += 1
                                        # Penalize aspiration of both worker and firm
                                        jobs.loc[network_index[k], 'aspir'] = max(0, jobs['aspir'][network_index[k]] - 0.05)
                                        workers.loc[i, 'aspir'] = max(0, workers['aspir'][i] - 0.05)
                                    
                                    # Check if there are still jobs to apply
                                    if k < m-1:
                                        # Move on to the next job in the network
                                        k += 1
                                    else:
                                        # Update the count of failed matches
                                        match_fail += 1
                                        # Update the no jobs left indicator
                                        none_left = True
                                        # Add a record to the log
                                        log.append(str(round(compat,2)) + ': unmatched ' + str(i) + ' failed to match')

                            # --------------------------------
                            # For a matched job
                            # --------------------------------
                            
                            # Check if the job is matched
                            elif jobs['match'][network_index[k]] != 9999:

                                # Check if compatibility is greater than aspiration for both agents
                                if (compat > aspir_i) and (compat > jobs['compat'][network_index[k]]):
                                    # Match occurs
                                    
                                    # Let go a worker who is currently occupying the job
                                    let_go = jobs['match'][network_index[k]]
                                    workers.loc[let_go, 'compat'] = 0
                                    workers.loc[let_go, 'match'] = 9999

                                    # Update compatibility levels
                                    workers.loc[i, 'compat'] = compat
                                    jobs.loc[network_index[k], 'compat'] = compat

                                    # Update match status
                                    workers.loc[i, 'match'] = network_index[k]
                                    jobs.loc[network_index[k], 'match'] = i

                                    # Update required education and skill
                                    workers.loc[i, 'r_edu'] = jobs['edu'][network_index[k]]
                                    workers.loc[i, 'r_skill'] = jobs['skill'][network_index[k]]
                                    
                                    # Add a record to the log
                                    log.append(str(round(compat, 2)) + ': unmatched ' + str(i) + ' matched with matched ' + str(network_index[k]))
                                    
                                    # Update the count of unmatched workers matching with matched jobs
                                    uw_mj += 1

                                    # Update the match indicator
                                    match = True

                                    # Update the all matched indicator
                                    all_matched = (9999 not in np.unique(workers['match']))
                                else:
                                    # Match fails
                                    
                                    # Check if the worker rejects the job
                                    if (compat <= aspir_i) and (compat > jobs['compat'][network_index[k]]):
                                        # Update the count of rejections by the workers
                                        w_rejects += 1 
                                        # Penalize aspiration of the firm
                                        jobs.loc[network_index[k], 'aspir'] = max(0, jobs['aspir'][network_index[k]] - 0.05)
                                    
                                    # Check if the firm rejects the worker
                                    elif (compat > aspir_i) and (compat <= jobs['compat'][network_index[k]]):
                                        # Update the count of rejections by the firms
                                        j_rejects += 1
                                        # Penalize aspiration of the worker
                                        workers.loc[i, 'aspir'] = max(0, workers['aspir'][i] - 0.05)

                                    # Check if both worker and firm reject each other
                                    elif (compat <= aspir_i) and (compat <= jobs['compat'][network_index[k]]):
                                        # Update the count of mutual rejections
                                        mututal_reject += 1
                                        # Penalize aspiration of both worker and firm
                                        jobs.loc[network_index[k], 'aspir'] = max(0, jobs['aspir'][network_index[k]] - 0.05)
                                        workers.loc[i, 'aspir'] = max(0, workers['aspir'][i] - 0.05)
                                    
                                    # Check if there are still jobs to apply
                                    if k < m-1:
                                        # Move on to the next job in the network
                                        k += 1
                                    else:
                                        # Update the count of failed matches
                                        match_fail += 1
                                        # Update the no jobs left indicator
                                        none_left = True
                                        # Add a record to the log
                                        log.append(str(round(compat,2)) + ': unmatched ' + str(i) + ' failed to match')
                    
                    # --------------------------------
                    # For a matched worker
                    # --------------------------------

                    # Check if the worker is matched
                    elif workers['match'][i] != 9999:
                        
                        # Pull the workers' current jobs
                        current_job = workers['match'][i]

                        # Generate the size of network
                        m = round(np.random.normal(n / 2, n / 10))
                        if m < 1:
                            m = 1
                        elif m > n - 1:
                            m = n - 1

                        # List for network jobs    
                        network_index = []

                        # Check if the number of empty jobs is 10% or less
                        empty_jobs = 0
                        if jobs['match'].value_counts()[9999] <= 0.1 * n:
                            for empty_job in jobs.loc[jobs['match'] == 9999].index:
                                # Add empty jobs to the network
                                network_index.append(empty_job)
                                # Penalize the aspiration level of the empty job
                                jobs.loc[empty_job, 'aspir'] = max(0, jobs['aspir'][empty_job] - 0.05)
                                # Update the count of empty jobs
                                empty_jobs += 1

                        j = 0
                        # Check if there is space left in the network
                        while j < m - empty_jobs:
                            # Randomly pick a job from the population
                            k = random.randint(0, n - 1)
                            # Check if the job is not already in the network
                            if k not in network_index:
                                # Add the job to the network
                                network_index.append(k)
                                j += 1

                        # Pull the data from the jobs dataset corresponding to the network
                        network = jobs.loc[network_index]

                        # Sort jobs in network by wage
                        network = network.sort_values(['match', 'wage'], ascending=False)

                        # Generate a list of indicies from the jobs pool dataset
                        network_index = network.index

                        # Indicator for a match occuring
                        match = False

                        # Indicator for being rejected by all jobs in the network
                        none_left = False

                        # Loop through all jobs in network until either a match occurs or no unapplied jobs left
                        k = 0
                        while (match == False) and (none_left == False):

                            # !!! potentially redundant !!!
                            # print(str(i) + ' applies for ' + str(network_index[k]) + ', number ' + str(k))

                            # Pull the agents' characteristics from the dataset
                            aspir_i = workers['aspir'][i]
                            aspir_j = jobs['aspir'][network_index[k]]
                            edu_w = workers['edu'][i]
                            edu_j = jobs['edu'][network_index[k]]
                            skill_w = workers['skill'][i]
                            skill_j = jobs['skill'][network_index[k]]

                            # Compute agents' compatibility
                            compat = compatibility_distance(edu_w, edu_j, 4, skill_w, skill_j, 500)
                            
                            # --------------------------------
                            # For an unmatched job
                            # --------------------------------
                            
                            # Check if the job is unmatched
                            if jobs['match'][network_index[k]] == 9999:

                                # Check if compatibility is greater than aspiration for both agents
                                if (compat > jobs['compat'][network_index[k]]) and (compat > aspir_j):
                                    # Match occurs

                                    # Quit worker's current job
                                    jobs.loc[jobs['match'] == i, 'compat'] = 0
                                    jobs.loc[jobs['match'] == i, 'match'] = 9999

                                    # Update compatibility levels
                                    workers.loc[i, 'compat'] = compat
                                    jobs.loc[network_index[k], 'compat'] = compat
                                    
                                    # Update match status
                                    workers.loc[i, 'match'] = network_index[k]
                                    jobs.loc[network_index[k], 'match'] = i

                                    # Update required education and skill
                                    workers.loc[i, 'r_edu'] = jobs['edu'][network_index[k]]
                                    workers.loc[i, 'r_skill'] = jobs['skill'][network_index[k]]
                                    
                                    # Add a record to the log
                                    log.append(str(round(compat,2)) + ': matched ' + str(i) + ' matched with unmatched ' + str(network_index[k]))
                                    
                                    # Update the count of matched workers matching with unmatched jobs
                                    mw_uj += 1

                                    # Update the match indicator
                                    match = True
                                    
                                    # Update the all matched indicator
                                    all_matched = (9999 not in np.unique(workers['match']))
                                else:
                                    # Match fails

                                    # Check if the worker rejects the job
                                    if (compat <= jobs['compat'][network_index[k]]) and (compat > aspir_j):
                                        # Update the count of rejections by the workers
                                        w_rejects += 1
                                        # Penalize aspiration of the firm
                                        jobs.loc[network_index[k], 'aspir'] = max(0, jobs['aspir'][network_index[k]] - 0.05)
                                    
                                    # Check if the firm rejects the worker
                                    elif (compat > jobs['compat'][network_index[k]]) and (compat <= aspir_j):
                                        # Update the count of rejections by the firms
                                        j_rejects += 1
                                        # Penalize aspiration of the worker
                                        workers.loc[i, 'aspir'] = max(0, workers['aspir'][i] - 0.05)
                                    
                                    # Check if both worker and firm reject each other
                                    elif (compat <= jobs['compat'][network_index[k]]) and (compat <= aspir_j):
                                        # Update the count of mutual rejections
                                        mututal_reject += 1
                                        # Penalize aspiration of both worker and firm
                                        jobs.loc[network_index[k], 'aspir'] = max(0, jobs['aspir'][network_index[k]] - 0.05)
                                        workers.loc[i, 'aspir'] = max(0, workers['aspir'][i] - 0.05)
                                    
                                    # Check if there are still jobs to apply
                                    if k < m-1:
                                        # Move on to the next job in the network
                                        k += 1
                                    else:
                                        # Update the count of failed matches
                                        match_fail += 1
                                        # Update the no jobs left indicator
                                        none_left = True
                                        # Add a record to the log
                                        log.append(str(round(compat,2)) + ': matched ' + str(i) + ' failed to match')
                            
                            # --------------------------------
                            # For a matched job
                            # --------------------------------

                            # Check if the job is matched
                            elif jobs['match'][network_index[k]] != 9999:
                                
                                # Check if compatibility is greater than aspiration for both agents
                                if (compat > jobs['compat'][network_index[k]]) and (compat > jobs['compat'][network_index[k]]):
                                    # Match occurs
                                    
                                    # Quit worker's current job
                                    jobs.loc[jobs['match'] == i, 'compat'] = 0
                                    jobs.loc[jobs['match'] == i, 'match'] = 9999
                                    
                                    # Let go a worker who is currently occupying the job
                                    let_go = jobs['match'][network_index[k]]
                                    workers.loc[let_go, 'compat'] = 0
                                    workers.loc[let_go, 'match'] = 9999

                                    # Update compatibility levels
                                    workers.loc[i, 'compat'] = compat
                                    jobs.loc[network_index[k], 'compat'] = compat

                                    # Update match status
                                    workers.loc[i, 'match'] = network_index[k]
                                    jobs.loc[network_index[k], 'match'] = i

                                    # Update required education and skill
                                    workers.loc[i, 'r_edu'] = jobs['edu'][network_index[k]]
                                    workers.loc[i, 'r_skill'] = jobs['skill'][network_index[k]]
                                    
                                    # Add a record to the log
                                    log.append(str(round(compat,2)) + ': matched ' + str(i) + ' matched with matched ' + str(network_index[k]))
                                    
                                    # Update the count of matched workers matching with matched jobs
                                    mw_mj += 1

                                    # Update the match indicator
                                    match = True

                                    # Update the all matched indicator
                                    all_matched = (9999 not in np.unique(workers['match']))
                                else:
                                    # Match fails

                                    # Check if the worker rejects the job
                                    if (compat <= jobs['compat'][network_index[k]]) and (compat > jobs['compat'][network_index[k]]):
                                        # Update the count of rejections by the workers
                                        w_rejects += 1
                                        # Penalize aspiration of the firm
                                        jobs.loc[network_index[k], 'aspir'] = max(0, jobs['aspir'][network_index[k]] - 0.05)
                                    
                                    # Check if the firm rejects the worker
                                    elif (compat > jobs['compat'][network_index[k]]) and (compat <= jobs['compat'][network_index[k]]):
                                        # Update the count of rejections by the firms
                                        j_rejects += 1
                                        # Penalize aspiration of the worker
                                        workers.loc[i, 'aspir'] = max(0, workers['aspir'][i] - 0.05)

                                    # Check if both worker and firm reject each other
                                    elif (compat <= jobs['compat'][network_index[k]]) and (compat <= jobs['compat'][network_index[k]]):
                                        # Update the count of mutual rejections
                                        mututal_reject += 1
                                        # Penalize aspiration of both worker and firm
                                        jobs.loc[network_index[k], 'aspir'] = max(0, jobs['aspir'][network_index[k]] - 0.05)
                                        workers.loc[i, 'aspir'] = max(0, workers['aspir'][i] - 0.05)

                                    # Check if there are still jobs to apply
                                    if k < m-1:
                                        # Move on to the next job in the network
                                        k += 1
                                    else:
                                        # Update the count of failed matches
                                        match_fail += 1
                                        # Update the no jobs left indicator
                                        none_left = True
                                        # Add a record to the log
                                        log.append(str(round(compat,2)) + ': matched ' + str(i) + ' failed to match')
            
            # !!! potentially redundant !!!
            """
            if iterat > 0:
                log_df = {'record': log}
                log_df = pd.DataFrame(log)
                log_df.to_csv('log' + spec +'.csv')
                workers.to_csv('workers' + spec +'.csv')
                jobs.to_csv('jobs' + spec +'.csv')
            """

        # !!! potentially redundant !!!
        """
        mean = np.mean(workers['compat'])
        sd = np.std(workers['compat'])
        workers['well_matched'] = np.random.beta(5 - (workers['compat'] - mean) / sd / 2,
                                                5 + (workers['compat'] - mean) / sd / 2)
        workers['well_matched'] = round(workers['well_matched'])
        """

        # --------------------------------------
        # Calculating output of mismatch measures
        # --------------------------------------

        # Indirect Self-Assessment (ISA)

        # Calculate ISA output
        conditions = [workers['edu'] > workers['r_edu'],
                    workers['edu'] == workers['r_edu'],
                    workers['edu'] < workers['r_edu']]
        values = [1,
                0,
                -1]
        workers['isa'] = np.select(conditions, values, default=math.nan)

        # Realised Matches (RM)

        # RM classification threshold
        SDs = 1

        # Calculate occupation-specific mode of education
        conditions = []
        values = []
        for group in workers['occupat'].unique():
            # Calculate the mode
            mode_edu = st.mode(workers.loc[workers['occupat'] == group, 'edu'])
            # Append occupation group to the conditions list
            conditions.append((workers['occupat'] == group))
            # Append mode of education to the values list
            values.append(mode_edu)
        workers['mode_edu'] = np.select(conditions, values, default=math.nan)

        # Calculate occupation-specific standard deviation of education
        conditions = []
        values = []
        for group in workers['occupat'].unique():
            # Calculate the standard deviation
            std_edu = workers.loc[workers['occupat'] == group, 'edu'].std()
            # Append occupation group to the conditions list
            conditions.append((workers['occupat'] == group))
            # Append standard deviation of education to the values list
            values.append(std_edu)
        workers['sd_edu'] = np.select(conditions, values, default=math.nan)

        # Calculate RM output
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

        # Pellizzari-Fichen (PF)

        # Well-matched indicator
        conditions = [workers['compat'] < well_matched_threshold,
                    workers['compat'] >= 0.7229727]
        values = [0,
                1]
        workers['well_matched'] = np.select(conditions, values, default=math.nan)

        # Calculate PF lower bound (occupation-specific 5th percentile of skill of well-matched workers)
        conditions = []
        values = []
        for group in workers['occupat'].unique():
            # Calculate the 5th percentile of skill of well-matched workers
            sm_min = workers.loc[
                (workers['occupat'] == group) * (workers['well_matched'] == 1) == 1, 'skill'].quantile(0.05)
            # Append occupation group to the conditions list
            conditions.append((workers['occupat'] == group))
            # Append 5th percentile of skill to the values list
            values.append(sm_min)
        workers['pf_min'] = np.select(conditions, values, default=math.nan)

        # Calculate PF upper bound (occupation-specific 95th percentile of skill of well-matched workers)
        conditions = []
        values = []
        for group in workers['occupat'].unique():
            # Calculate the 95th percentile of skill of well-matched workers
            sm_max = workers.loc[
                (workers['occupat'] == group) * (workers['well_matched'] == 1) == 1, 'skill'].quantile(0.95)
            # Append occupation group to the conditions list
            conditions.append((workers['occupat'] == group))
            # Append 95th percentile of skill to the values list
            values.append(sm_max)
        workers['pf_max'] = np.select(conditions, values, default=math.nan)

        # Calculate PF output
        conditions = [
            (workers['skill'] < workers['pf_min']),
            ((workers['skill'] < workers['pf_max']) & (workers['skill'] >= workers['pf_min'])),
            (workers['skill'] >= workers['pf_max'])]
        values = [
            -1,
            0,
            1]
        workers['pf'] = np.select(conditions, values, default=math.nan)

        # --------------------------------------
        # Post-simulation stats update
        # --------------------------------------

        # Calculate average aspiration levels after the simulation
        av_aspir_w2 = workers['aspir'].mean()

        # Calculate average aspiration levels after the simulation
        av_aspir_j2 = jobs['aspir'].mean()

        # Calculate average compatibility levels after the simulation
        av_compat_fin = workers['compat'].mean()

        # !!! potentially redundant !!!
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
            # Calculate average aspiration levels for workers after the first iteration
            av_compat_1 = workers['compat'].mean()

        # New record for the simulation stats dataframe
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

        # !!! potentially redundant !!!
        """
        stats.to_csv('stats' + spec +'.csv')
        """
        
        # Update the rep counter
        rep += 1

    # Save the data
    workers.to_csv('workers' + spec +'.csv')
    jobs.to_csv('jobs' + spec +'.csv')
    stats.to_csv('stats' + spec +'.csv')

    # Data saving message
    print()
    print('File workers' + spec +'.csv is saved to ' + os.getcwd())
    print('File jobs' + spec +'.csv is saved to ' + os.getcwd())
    print('File stats' + spec +'.csv is saved to ' + os.getcwd())

    # Post-simulation stats message
    print()
    print('---------------------------------------')
    print('STATS')
    print('---------------------------------------')
    print(stats)
    print('---------------------------------------')

    return workers, jobs, stats, log