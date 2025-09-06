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
