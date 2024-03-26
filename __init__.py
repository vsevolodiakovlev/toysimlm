"""
Toy microsimulation of labour mismatch.

This module implements a simple microsimulation of labour mismatch based on a static version of the two-sided matching model 
developed by Zinn (2012). The purpose of the simulation is to produce matching outcomes of a labour market populated by 
a set of workers and a set of jobs with pre-defined characteristics and compute the output of various mismatch measures. 
This can then be analysed to gain a deeper understanding of the construct of the mismatch measures.
For more information, see https://github.com/vsevolodiakovlev/toysimlm/blob/main/README.md.

The main function of the simulation is `simulate`. To learn more about the simulation parameters, run `help(toysimlm.simulate)`.

The module also contains two helper functions: `generate_random_sequence` and `compatibility_distance`.
Both functions are used in the `simulate` function and do not need to be called separately.

author: Vsevolod Iakovlev
email: vsevolod.v.iakovlev@gmail.com
"""

from .src.main import simulate
from .src.main import generate_random_sequence
from .src.main import compatibility_distance
