from celosia.hdl import HDL
from celosia.hdl.vhdl.backend import VHDLModule
from typing import Union
from amaranth.hdl import _ast, _ir

class VHDL(HDL):
    ModuleClass = VHDLModule

    submodules_first = True
    extensions = ['vhd', 'vhdl']
    open_comment = '-- '
    close_comment = ''

    def __init__(self, spaces: int = 4, std_logic: dict[str, dict[str, list[str]]] = None):
        self.spaces = spaces
        self.std_logic = std_logic or {}

        if not isinstance(self.std_logic, dict):
            raise ValueError(f"Invalid std_logic received, expected a dictionary, but got: {self.std_logic}")

    def set_module_params(self, module: VHDLModule):
        if isinstance(module, VHDLModule):
            module.add_std_logic(**self.std_logic)

def convert(
    elaboratable: Union[_ir.Elaboratable, _ir.Fragment, _ir.Design],
    name: str = 'top',
    ports: list[_ast.Signal] = None,
    platform = None,
    spaces: int = 4,
    std_logic: dict[str, dict[str, list[str]]] = None,
    **kwargs
):
    return VHDL(
        spaces = spaces,
        std_logic = std_logic,
    ).convert(
        elaboratable,
        name = name,
        platform = platform,
        ports = ports,
        **kwargs
    )