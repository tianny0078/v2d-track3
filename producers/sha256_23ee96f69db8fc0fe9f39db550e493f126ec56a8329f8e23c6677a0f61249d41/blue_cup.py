"""blue_cup: tapered open cup, see cup.py for the model and how each parameter is estimated."""
from cup import fit_cup, build_cup
RGBA = '0.30 0.42 0.62 1' if 'blue_cup' == 'blue_cup' else '0.85 0.80 0.68 1'
def fit(base): return fit_cup(base)
def build(p, res='high'): return build_cup(p, res)
