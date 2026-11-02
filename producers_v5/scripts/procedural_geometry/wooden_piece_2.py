"""wooden_piece_2: slotted bamboo board, see board.py (slot side = the edge the SAM3D gap reaches)."""
from board import fit_board, build_board, RGBA
def fit(base): return fit_board(base)
def build(p, res='high'): return build_board(p, res)
