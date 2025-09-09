from celosia.hdl import HDL
from celosia.hdl.vhdl.backend import VHDLModule
from typing import Union
from amaranth.hdl import _ast, _ir

class VHDL(HDL):
    ModuleClass = VHDLModule

    top_first = False
    extensions = ['vhd', 'vhdl']
    open_comment = '-- '
    close_comment = ''

    def __init__(self, spaces: int = 4, blackboxes: list[dict[str, Union[int, str, tuple]]] = None):
        self.spaces = spaces
        self.blackboxes = blackboxes

        if blackboxes:
            raise NotImplementedError("VHDL blackboxes not supported")

def convert(
    elaboratable: Union[_ir.Elaboratable, _ir.Fragment, _ir.Design],
    name: str = 'top',
    ports: list[_ast.Signal] = None,
    platform = None,
    spaces: int = 4,
    blackboxes: list[dict[str, Union[int, str, tuple]]] = None,
):
    return VHDL(
        spaces = spaces,
        blackboxes = blackboxes,
    ).convert(
        elaboratable,
        name = name,
        platform = platform,
        ports = ports,
    )