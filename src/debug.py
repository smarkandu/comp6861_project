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