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

    case_sensitive = False
    top_first = True
    open_comment = ''
    close_comment = ''

    def __init__(self, spaces: int = 4):
        self.spaces = spaces

    @property
    def default_extension(self) -> str:
        return type(self).default_extension

    @classmethod
    def _change_case(cls, name: str) -> str:
        return name if cls.case_sensitive else name.lower()

    def convert(self, elaboratable: Any, name='top', platform=None, ports=None, **kwargs):
        from amaranth.back import verilog, rtlil

        def _convert_rtlil_text(rtlil_text, *args, **kwargs):
            return rtlil_text

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

        try:
            verilog._convert_rtlil_text = _convert_rtlil_text
            rtlil.Emitter.indent = indent
            rtlil.Module = self.ModuleClass
            rtlil._const = self.ModuleClass._const
            return verilog.convert(elaboratable, name=name, ports=ports, platform=platform, **kwargs)
        finally:
            verilog._convert_rtlil_text = orig__convert_rtlil_text
            rtlil.Emitter.indent = orig__emitter_indent
            rtlil.Module = orig__module
            rtlil._const = orig__const

def get_lang_map() -> dict[str, type[HDL]]:
    lang_map: dict[str, type[HDL]] = {}
    for _, name, _ in pkgutil.iter_modules(__path__, __name__ + '.'):
        for HDLType in importlib.import_module(name).__dict__.values():
            if isinstance(HDLType, type) and issubclass(HDLType, HDL):
                lang_map[name.split('.')[-1]] = HDLType
    return lang_map