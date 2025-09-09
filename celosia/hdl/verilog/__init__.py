from celosia.hdl import HDL
from celosia.hdl.verilog.backend import VerilogModule
from typing import Union
from amaranth.hdl import _ast, _ir

class Verilog(HDL):
    ModuleClass = VerilogModule

    top_first = False
    extensions = ['v']
    open_comment = '/* '
    close_comment = ' */'

def convert(
    elaboratable: Union[_ir.Elaboratable, _ir.Fragment, _ir.Design],
    name: str = 'top',
    ports: list[_ast.Signal] = None,
    platform = None,
    spaces: int = 4,
    blackboxes: list[dict[str, Union[int, str, tuple]]] = None,
):
    return Verilog(
        spaces = spaces,
        blackboxes = blackboxes,
    ).convert(
        elaboratable,
        name = name,
        ports = ports,
        platform = platform,
    )