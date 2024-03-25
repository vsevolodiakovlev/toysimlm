"""
Toy microsimulation of labour mismatch.

The main function of the simulation is `simulate`. To learn more about the simulation parameters, run `help(toysimlm.simulate)`.

The module also contains two helper functions: `generate_random_sequence` and `compatibility_distance`.
Both functions are used in the `simulate` function and do not need to be called separately.

author: Vsevolod Iakovlev
email: vsevolod.v.iakovlev@gmail.com
"""

from .src.main import simulate
from .src.main import generate_random_sequence
from .src.main import compatibility_distance
