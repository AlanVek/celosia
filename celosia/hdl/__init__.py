from typing import Any
import importlib
import pkgutil
from contextlib import contextmanager
from celosia.hdl.module import Module
from amaranth.hdl import _ast

# Overrides
#####################################################################################################
from amaranth.back import verilog, rtlil
from amaranth.back.verilog import _convert_rtlil_text
from amaranth.back.rtlil import Module as rtlil_Module, Emitter, _const, Design
from amaranth.hdl import _ir, _nir
from amaranth.hdl._ir import _compute_ports, _compute_io_ports, _add_name
#####################################################################################################

class HDLExtensions(type):
    extensions: list[str] = []

    @property
    def default_extension(self) -> str:
        if not self.extensions:
            raise RuntimeError(f"No extensions defined for {self.__class__.__name__}")
        return self.extensions[0]

class HDL(metaclass=HDLExtensions):
    ModuleClass = Module

    top_first = True
    open_comment = ''
    close_comment = ''

    def __init__(self, spaces: int = 4):
        self.spaces = spaces

    @property
    def default_extension(self) -> str:
        return type(self).default_extension

    def generate_overrides(self):
        def _new_add_name(assigned_names: set[str], name: str) -> str:
            name = self.ModuleClass.filter_name(name, assigned_names=assigned_names)
            assigned_names.add(name)
            return name

        def _new_compute_ports(netlist: _nir.Netlist):
            ret = _compute_ports(netlist)
            for module in netlist.modules:
                assigned = module.signal_names.values() | module.ports.keys() | module.io_ports.keys()
                for name in tuple(module.ports.keys()):
                    if name.startswith('port$'):
                        new = self.ModuleClass.filter_name(name, assigned)
                        module.ports[new] = module.ports.pop(name)
                        assigned.add(new)
            return ret

        def _new_compute_io_ports(netlist: _nir.Netlist, ports):
            ret = _compute_io_ports(netlist, ports)
            for module in netlist.modules:
                assigned = module.signal_names.values() | module.ports.keys() | module.io_ports.keys()
                for name in tuple(module.io_ports.keys()):
                    if name.startswith('port$'):
                        new = self.ModuleClass.filter_name(name, assigned)
                        module.io_ports[new] = module.io_ports.pop(name)
                        assigned.add(new)
            return ret

        def _new_convert_rtlil_text(rtlil_text, *args, **kwargs):
            return rtlil_text

        def _new_to_str(design: rtlil.Design, *args, **kwargs):
            emitter = rtlil.Emitter()

            modules = design.modules.values()
            if self.ModuleClass.submodules_first:
                modules = reversed(modules)
            for module in modules:
                module.emit(emitter)
            return str(emitter)

        @contextmanager
        def _new_indent(emitter: rtlil.Emitter):
            orig = emitter._indent
            emitter._indent += ' ' * self.spaces
            yield
            emitter._indent = orig

        verilog._convert_rtlil_text = _new_convert_rtlil_text
        rtlil.Emitter.indent = _new_indent
        rtlil.Module = self.ModuleClass
        rtlil._const = self.ModuleClass._const
        rtlil.Design.__str__ = _new_to_str
        _ir._add_name = _new_add_name
        _ir._compute_ports = _new_compute_ports
        _ir._compute_io_ports = _new_compute_io_ports

    def cleanup_overrides(self):
        verilog._convert_rtlil_text = _convert_rtlil_text
        rtlil.Emitter.indent = Emitter.indent
        rtlil.Module = rtlil_Module
        rtlil._const = _const
        rtlil.Design.__str__ = Design.__str__
        _ir._add_name = _add_name
        _ir._compute_ports = _compute_ports
        _ir._compute_io_ports = _compute_io_ports

    def convert(self, elaboratable: Any, name='top', platform=None, ports=None, **kwargs):
        try:
            self.generate_overrides()
            return verilog.convert(elaboratable, name=name, ports=ports, platform=platform, **kwargs)
        finally:
            self.cleanup_overrides()

def get_lang_map() -> dict[str, type[HDL]]:
    lang_map: dict[str, type[HDL]] = {}
    for _, name, _ in pkgutil.iter_modules(__path__, __name__ + '.'):
        for HDLType in importlib.import_module(name).__dict__.values():
            if isinstance(HDLType, type) and issubclass(HDLType, HDL):
                lang_map[name.split('.')[-1]] = HDLType
    return lang_map