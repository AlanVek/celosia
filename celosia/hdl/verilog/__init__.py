from celosia.hdl import HDL
from celosia.hdl.verilog.backend import VerilogModule
from typing import Union
from amaranth.hdl import _ast, _ir

class Verilog(HDL):
    ModuleClass = VerilogModule

    submodules_first = False
    extensions = ['v']
    open_comment = '/* '
    close_comment = ' */'

def convert(
    elaboratable: Union[_ir.Elaboratable, _ir.Fragment, _ir.Design],
    name: str = 'top',
    ports: list[_ast.Signal] = None,
    platform = None,
    spaces: int = 4,
    **kwargs
):
    return Verilog(
        spaces = spaces,
    ).convert(
        elaboratable,
        name = name,
        ports = ports,
        platform = platform,
        **kwargs
    )