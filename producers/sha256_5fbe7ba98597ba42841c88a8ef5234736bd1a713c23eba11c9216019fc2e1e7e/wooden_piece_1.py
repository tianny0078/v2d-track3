"""wooden_piece_1: slotted bamboo board, see board.py. The SAM3D gap crosses the whole board; the slot is on the +s half
of the slot line (canonical +x), from the ep25 frame-0/30 image check documented in board.py."""
from board import fit_board, build_board, RGBA
def fit(base): return fit_board(base, slot_side='max')
def build(p, res='high'): return build_board(p, res)
