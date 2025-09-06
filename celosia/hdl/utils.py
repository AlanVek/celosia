import re
from typing import Optional

def const_params(value: str, ret_idx=False) -> tuple[int, int, Optional[int]]:
    ret = None
    const_match = re.match(r"(\d+)'([\d|-]+)", value)
    if const_match is not None:
        width = int(const_match.group(1))
        value = int(const_match.group(2), 2)
        if ret_idx:
            ret = (width, value, const_match.end())
        else:
            ret = (width, value)
    return ret

def slice_params(value: str, ret_idx=False) -> tuple[str, int, int, Optional[int]]:
    ret = None

    slice_pattern = re.compile(r'(.*?) \[(.*?)\]')
    slice_match = slice_pattern.match(value)
    if slice_match is not None:
        name, index = slice_match.groups()

        if ':' in index:
            stop, start = map(int, index.split(':'))
        else:
            stop = start = int(index)

        if ret_idx:
            ret = (name, start, stop, slice_match.end())
        else:
            ret = (name, start, stop)

    return ret