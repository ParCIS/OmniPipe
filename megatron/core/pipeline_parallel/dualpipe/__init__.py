from ..chimera import init_chimera_parallel_state as init_dualpipe_parallel_state
from .dualpipe_scheduler import DualPipeScheduler

DUALPIPE_SCHEDULER = None
def get_dualpipe_forward_backward_func():
    global DUALPIPE_SCHEDULER
    if DUALPIPE_SCHEDULER is None:
        DUALPIPE_SCHEDULER = DualPipeScheduler()
    return DUALPIPE_SCHEDULER

__all__ = ["init_dualpipe_parallel_state", "get_dualpipe_forward_backward_func"]

