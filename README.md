# Toy Microsimulation of Labour Mismatch

[Usage](#to-use-the-module) | [Examples](#other-examples) | [Codebook](#codebook) | [References](#references)

This python module implements a simple microsimulation of labour mismatch based on a static version of the two-sided matching model developed by Zinn (2012). The purpose of the simulation is to produce matching outcomes of a labour market populated by a set of workers and a set of jobs with pre-defined characteristics and compute the output of multiple mismatch measures, which can then be analysed to gain a deeper understanding of their construct. For the description of the economic model, available mismatch measures and a demonstration, please see the [slides](./toysimlm_slides.pdf).

Available mismatch measures:
- Realised Matches (RM)
- Indirect Self-Assessment (ISA)
- Pellizzari and Fichen (2017) (PF)

*Please note that this microsimulation is based on a stylized **(toy)** model and should be taken as more of a methodological experiment and illustration than an inference tool.*

*Any suggestions regarding either the code or economic model are most welcome!*

![Light mode figure](./figures/figure_light.jpg#gh-light-mode-only)
![Dark mode figure](./figures/figure_dark.jpg#gh-dark-mode-only)


## To use the module

1. Create a new directory where you want the simulation output files to be generated and make it your working directory
2. Clone the repository to that directory
3. Make sure ``pandas`` package is installed
4. Import the module:
```python
import toysimlm
```
5. Type ``help(toysimlm)`` to view the module's description or ``help(toysimlm.simulate)`` for the simulation parameters
6. Run the simulation with default parameters:
```python
stats, workers, jobs, log = toysimlm.simulate()
```
7. This produces 4 CSV files:
 - **stats** — main output file containing the simulation statistics for each repetition, including the shares of under, well and over-matched workers according to 3 different measures
 - **workers** and **jobs** — additional files containing the corresponding characteristics in the last repetition that illustrate the pre-defined distributions 
 - **log** — additional file containing matching outcomes for each iteration of the matching procedure for the last repetition illustrating the algorithm
8. Analyse the simulation output using a statistical software of your choice

## Other examples

 - Run the simulation for 1000 reps instead of 100; with probability 0.25 increase workers' education by 1.8 (two standard deviations):
 ```python
stats, workers, jobs, log = toysimlm.simulate(reps = 1000,
                                               spec = '_n100x1000_edu_inc',
                                               w_treatment = True,
                                               w_treatment_prob = 0.25
                                               w_edu_treatment_mag = 1.8)
 ```

- Set the numbers of workers and firms to 200 each instead of 100;  with probability 0.5 decrease jobs' skill requirement by 92 (two standard deviations):
 ```python
stats, workers, jobs, log = toysimlm.simulate(n = 200,
                                               spec = '_n200x100_skillreq_dec',
                                               j_treatment = True,
                                               j_treatment_prob = 0.5
                                               j_skill_treatment_mag = -92)
 ``` 

- Run the simulation while saving the data after each iteration and every time a worker is taken through the matching procedure (as opposed to upon the completion). If either ``n`` and ``reps`` are large, this may increase the completion time of the simulation considerably but could be useful for troubleshooting.
 ```python
stats, workers, jobs, log = toysimlm.simulate(sim_rep_updates = True,
                                               worker_updates = True)
 ```

## Codebook

### Stats
- **iterat** - number of iterations
- **uw_uj** - unmatched workers matching with unmatched job
- **uw_mj** - unmatched workers matching with matched job
- **mw_uj** - matched workers matching with unmatched job
- **mw_mj** - matched workers matching with matched job
- **match_fail** - failed matches
- **w_rejects** - rejections by the workers
- **j_rejects** - rejections by the firms
- **mututal_reject** - mutual rejections
- **av_aspir_w1** - average aspiration level for workers before the simulation
- **av_aspir_w2** - average aspiration level for workers after the simulation
- **av_aspir_j1** - average aspiration level for firms before the simulation
- **av_aspir_j2** - average aspiration level for firms after the simulation
- **av_compat_1** - average compatibility level for workers after the first iteration (zero in the first iteration)
- **av_compat_fin** - average compatibility level for workers after the simulation
- **w_edu** - average workers' education
- **w_skill** - average workers' skill
- **occupat_0** - number of workers in occupation 0
- **occupat_1** - number of workers in occupation 1
- **occupat_2** - number of workers in occupation 2
- **well_matched** - well-matched indicator
- **rm_under** - share of under-matched workers according to RM
- **rm_well** - share of well-matched workers according to RM
- **rm_over** - share of over-matched workers according to RM
- **isa_under** - share of under-matched workers according to ISA
- **isa_well** - share of well-matched workers according to ISA
- **isa_over** - share of over-matched workers according to ISA
- **pf_under** - share of under-matched workers according to PF
- **pf_well** - share of well-matched workers according to PF
- **pf_over** - share of over-matched workers according to PF
- **j_edu** - average firms' education requirement
- **j_skill** - average firms' skill requirement
- **j_wage** - average wage
- **w_treated** - share of treated workers
- **j_treated** - share of treated firms

### Workers
- **edu** - Education level
- **skill** - Skill level
- **compat** - Compatibility level if matched and 0 if unmatched
- **match** - Firm's index if matched and 9999 if unmatched
- **r_edu** - Required education level if matched and 0 if unmatched
- **r_skill** - Required skill level if matched and 0 if unmatched
- **aspir** - Current aspiration level
- **aspir_init** - Initial aspiration level
- **occupat** - Skill-based occupation group classification 
- **treatment** - Treatment indicator (1: treated, 0: not treated)
- **isa** - Skill mismatch according to ISA (-1: under, 0: well, 1: over)
- **mode_edu** - Occupation-specific mode of education level
- **sd_edu** - Occupation-specific standard deviation of education level
- **rm** - Educational mismatch according to RM (-1: under, 0: well, 1: over)
- **well_matched** - Well-matched indicator
- **pf_min** - PF lower bound
- **pf_max** - Pf upper bound
- **pf** - Skill mismatch according to PF (-1: under, 0: well, 1: over)

### Jobs
- **edu** - Education requirement
- **skill** - Skill requirement
- **wage** - Wage
- **compat** - Compatibility level if matched and 0 if unmatched
- **match** - Worker's index if matched and 9999 if unmatched 
- **aspir** - Current aspiration level
- **aspir_init** - Initial aspiration level
- **treatment** - Treatment indicator (1: treated, 0: not treated)

## References

- Pellizzari, M. and Fichen, A. (2017). *A new measure of skill mismatch: theory and evidence from PIAAC.* IZA Journal of Labor Economics, 6(1):1–30.
- Zinn, S. (2012). *A mate-matching algorithm for continuous-time microsimulation models.* International journal of microsimulation, 5(1):31-51.
