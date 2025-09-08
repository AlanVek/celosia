from amaranth.hdl import _ir
from typing import Any
import importlib
import pkgutil
from contextlib import contextmanager
from celosia.hdl.backend import Module

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

    def convert(self, elaboratable: Any, name='top', platform=None, ports=None, **kwargs):
        from amaranth.back import verilog, rtlil

        def _convert_rtlil_text(rtlil_text, *args, **kwargs):
            return rtlil_text

        def _to_str(design, *args, **kwargs):
            emitter = rtlil.Emitter()

            modules = design.modules.values()
            if self.ModuleClass.submodules_first:
                modules = reversed(modules)
            for module in modules:
                module.emit(emitter)
            return str(emitter)

        @contextmanager
        def indent(emitter: rtlil.Emitter):
            orig = emitter._indent
            emitter._indent += ' ' * self.spaces
            yield
            emitter._indent = orig

        orig__convert_rtlil_text = verilog._convert_rtlil_text
        orig__emitter_indent = rtlil.Emitter.indent
        orig__module = rtlil.Module
        orig__const = rtlil._const
        orig__str__ = rtlil.Design.__str__

        try:
            verilog._convert_rtlil_text = _convert_rtlil_text
            rtlil.Emitter.indent = indent
            rtlil.Module = self.ModuleClass
            rtlil._const = self.ModuleClass._const
            rtlil.Design.__str__ = _to_str
            return verilog.convert(elaboratable, name=name, ports=ports, platform=platform, **kwargs)
        finally:
            verilog._convert_rtlil_text = orig__convert_rtlil_text
            rtlil.Emitter.indent = orig__emitter_indent
            rtlil.Module = orig__module
            rtlil._const = orig__const
            rtlil.Design.__str__ = orig__str__

def get_lang_map() -> dict[str, type[HDL]]:
    lang_map: dict[str, type[HDL]] = {}
    for _, name, _ in pkgutil.iter_modules(__path__, __name__ + '.'):
        for HDLType in importlib.import_module(name).__dict__.values():
            if isinstance(HDLType, type) and issubclass(HDLType, HDL):
                lang_map[name.split('.')[-1]] = HDLType
    return lang_map