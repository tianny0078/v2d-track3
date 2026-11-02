"""Exec a command with transparent huge pages disabled for it and all its descendants (prctl PR_SET_THP_DISABLE=41,
inherited across fork/execve). Reason (2026-10-31): with THP defrag=madvise and a fragmented 1 TB host, every 2 MB THP
fault of the 26 GB replay snapshot hit direct compaction (~4 compaction stalls/s, kernel time 99%), so the replay save
of a 2M-step milestone ran > 1 h and the segment hit its 7200 s timeout. No training code or config is changed.
Usage: python3 scripts/thp_off_exec.py CMD [ARGS...]"""
import ctypes, os, sys
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(41, 1, 0, 0, 0) != 0: raise OSError(ctypes.get_errno(), 'prctl(PR_SET_THP_DISABLE) failed')
os.execvp(sys.argv[1], sys.argv[1:])
