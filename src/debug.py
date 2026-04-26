import os
import random
import numpy as np
import torch

DEBUG = False
VERBOSE = 1   # 0 = silent, 1 = normal, 2 = debug

def get_debug():
    return DEBUG

def get_verbose():
    return VERBOSE

def set_debug(enabled: bool):
    global DEBUG
    DEBUG = enabled


def set_verbose(level: int):
    global VERBOSE
    VERBOSE = level


def debug_print(msg: str):
    if DEBUG:
        print(msg)


def vprint(msg: str, level: int=1):
    if VERBOSE >= level:
        print(msg)

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    vprint(f"[Seed] Set random seed to {seed}", level=2)