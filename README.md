# Toy Microsimulation of Labour Mismatch

[Usage](#to-use-the-module) | [References](#references)

This module implements a simple midcrosimulation of a labour mismatch based on a static version of the two-sided matching model developed by Zinn et al. (2012). The purpose of the simulation is to produce matching outcomes of a labour market populated by a set of workers and a set of jobs with pre-defined characteristics and compute the output of various mismatch measure. This can then be analysed to gain a deeper understandin of the construct of the mismatch measures. For the description of the economic model and a demonstration, please see the [slides](./toysimlm_slides.pdf).


**Please note that this microsimulation is based on a stylized *(toy)* model and should be taken as more of a methodological experiment and illustration than a inference tool. Nevertheless, any suggestion regarding either the code or economics are most welcome**

## To use the module

1. Create a new direcory where you want the simulation output files to be generated and make it your working directory
2. Clone the repository to that directory
3. Make sure ``pandas`` package is installed
4. Import the module:
```python
import toysimlm
```
5. Run the simulation with default parameters:
```python
stats, workers, jobs, log = toysimlm.simulate()
```
6. This produces 4 csv files:
 - **stats** — main output file contaiining the simulation statistics for each repetition, including the shares of under, well and over matched workers according to 3 different measures
 - **workers** and **jobs** — additional files containing the corresponding characteristics in the last repetition that illustrate the pre-defined distributions 
 - **log** — additional file containing matching outcomes for each iteration of the matching procedure for the last repetition illustrating the algorithm
7. Analyse the simulation output using a statistical software of your choice

## Examples

## References

Sabine Zinn et al. A mate-matching algorithm for continuous-time microsimulation models. International journal of microsimulation, 5(1):31–51, 2012.
