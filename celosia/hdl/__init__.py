from textwrap import indent
import celosia.backend.signal as celosia_signal
import celosia.backend.module as celosia_module
import celosia.backend.statement as celosia_statement
from amaranth.hdl import _ast, _ir, _nir
from typing import Union, Any
import importlib
import pkgutil

class HDLExtensions(type):
    extensions: list[str] = []

    @property
    def default_extension(self) -> str:
        if not self.extensions:
            raise RuntimeError(f"No extensions defined for {self.__class__.__name__}")
        return self.extensions[0]

class HDL(metaclass=HDLExtensions):
    case_sensitive = False
    portsep = ','
    top_first = True

    open_comment = ''
    close_comment = ''

    template = """{name}
{ports}{initials}{submodules}{blocks}{assignments}
"""

    def __init__(self, spaces: int = 4):
        self.spaces = spaces
        self.signal_features: dict[str, list] = {}
        self.submodule_features: dict[str, list] = {}

        self.module: celosia_module.Module = None
        self.invalid_names: set[str] = set()
        self.name_map: dict[_ast.SignalDict, tuple[str, ...]] = _ast.SignalDict()
        self.hierarchy: tuple[str, ...] = ()

    @property
    def default_extension(self) -> str:
        return type(self).default_extension

    def sanitize(self, name: str) -> str:
        return name

    @classmethod
    def _change_case(cls, name: str) -> str:
        return name if cls.case_sensitive else name.lower()

    def _sanitize_something(self, name: str, extra: set[str] = None) -> str:
        invalid = self.invalid_names.copy()
        if extra is not None:
            invalid.update(extra)
        if not name:
            name = 'unnamed'

        curr_num = ''
        curr_idx = len(name) - 1
        while curr_idx >= 0 and name[curr_idx].isnumeric():
            curr_num = name[curr_idx] + curr_num
            curr_idx -= 1

        if curr_num:
            idx = int(curr_num) + 1
            _name = name[:curr_idx+1]
        else:
            idx = 0
            _name = name

        while self._change_case(name) in invalid:
            name = f'{_name}{idx}'
            idx += 1

        return name

    def _sanitize_name(self, entry: Union[celosia_module.Module, celosia_signal.Signal], extra: set[str] = None):
        entry.name = self._sanitize_something(self.sanitize(entry.name), extra=extra)

    def _sanitize_type(self, entry: celosia_module.Module, extra: set[str] = None) -> None:
        entry.type = self._sanitize_something(self.sanitize(entry.type), extra=extra)

    @classmethod
    def _add_invalid(cls, invalid: set[str], entry: str):
        invalid.add(cls._change_case(entry))

    def _cleanup_names(self, module: celosia_module.Module):
        if module.top:
            self._sanitize_name(module)
            self._add_invalid(self.invalid_names, module.name)

        local: set[str] = set()

        # First, we find out how submodules want to name their ports
        for submodule in module.submodules:
            if isinstance(submodule, celosia_module.InstanceModule):
                continue

            repeated = set()
            for subport in submodule.ports:
                subport.set_alt_name(self._sanitize_something(self.sanitize(subport.name), extra=repeated))
                self._add_invalid(repeated, subport.name)

        # For toplevel, ports have priority
        for port in module.ports:
            if module.top:
                self._sanitize_name(port, extra=local)
            self._add_invalid(local, port.name)

        for submodule in module.submodules:
            self._sanitize_name(submodule, extra=local)
            self._add_invalid(local, submodule.name)

            if not isinstance(submodule, celosia_module.InstanceModule):
                self._sanitize_type(submodule, extra=local)
            self._add_invalid(self.invalid_names, submodule.type)

        for mapping in module.signals.values():
            if not isinstance(mapping, celosia_signal.Port):
                # Update remapped name to match original signal. We're assuming that the sync signal has
                # already been sanitized
                if isinstance(mapping, celosia_signal.RemappedSignal):
                    mapping.name = f'{module.signals[mapping.sync_signal].name}_next'

                self._sanitize_name(mapping, extra=local)

            self._add_invalid(local, mapping.name)

        for submodule in module.submodules:
            self._cleanup_names(submodule)

    @staticmethod
    def _build_netlist(fragment: Any, ports=(), *, name="top", all_undef_to_ff=False, platform=None, **kwargs) -> _ir.NetlistEmitter:
        if isinstance(fragment, _ir.Design):
            design = fragment
        elif isinstance(fragment, (_ir.Fragment, _ir.Elaboratable)):
            design = _ir.Fragment.get(fragment, platform).prepare(ports=ports, hierarchy=(name,), **kwargs)
        else:
            raise ValueError(f"Invalid fragment type: {type(fragment)}")

        netlist = _nir.Netlist()
        emitter = _ir.NetlistEmitter(netlist, design, all_undef_to_ff=all_undef_to_ff)
        emitter.emit_fragment(design.fragment, None)
        netlist.check_comb_cycles()
        netlist.resolve_all_nets()
        _ir._compute_net_flows(netlist)
        _ir._compute_ports(netlist)
        _ir._compute_ionet_dirs(netlist)
        _ir._compute_io_ports(netlist, design.ports)

        return emitter

    def convert(self, fragment: Any, name: str = 'top', ports: list[_ast.Signal] = None, platform=None):
        emitter = self._build_netlist(fragment, ports=ports, name=name, platform=platform)
        m = celosia_module.Module(name, emitter).prepare()

        self.name_map.clear()
        self.hierarchy = ()
        self.invalid_names.clear()
        self.invalid_names.add('')

        self._cleanup_names(m)

        return self._convert_module(m)

    def reset(self):
        self.signal_features.clear()
        self.submodule_features.clear()

    def _convert_module(self, module: celosia_module.Module) -> str:
        self.module = module

        self.reset()
        self.hierarchy = self.hierarchy + (module.name,)

        if self.module.empty:
            return ''

        ports, initials, assignments, blocks = self._generate_signals()
        submodules = self._generate_submodules()

        features = {
            key: '\n'.join(value) for entry in [self.signal_features, self.submodule_features] for key, value in entry.items()
        }

        formats = {
            'name': self.module.type,
            **{key: indent(('\n' if value else '') + value, self.tabs()) for key, value in {
                'ports': ports, 'initials': initials, 'assignments': assignments, 'blocks': blocks,
                'submodules': submodules, **features,
            }.items()},
        }
        formats['tabs'] = self.tabs()

        converted = self.template.format(**formats)
        res = []

        if self.top_first:
            res.append(converted)

        hierarchy = self.hierarchy
        for submodule in self.module.submodules:
            if isinstance(submodule, celosia_module.InstanceModule):
                continue
            self.hierarchy = hierarchy
            res.append(self._convert_module(submodule))

        if not self.top_first:
            res.append(converted)

        return '\n'.join(res)

    def _generate_port(self, mapping: celosia_signal.Signal) -> str:
        return ''

    def _generate_initial(self, mapping: celosia_signal.Signal) -> str:
        return ''

    def _generate_assignment(self, mapping: celosia_signal.Signal, statement: celosia_statement.Statement) -> str:
        return ''

    def _generate_block(self, mapping: celosia_signal.Signal)  -> str:
        return ''

    def _generate_signal_features(self, mapping: celosia_signal.Signal):
        return

    def _generate_signals(self) -> tuple[str, str, str, str]:
        ports: list[str] = []
        initials: list[str] = []
        assignments: list[str] = []
        blocks: list[str] = []

        for mapping in self.module.signals.values():
            if not len(mapping.signal):
                continue

            # TODO: Check memories
            self.name_map[mapping.signal] = self.hierarchy + (mapping.name,)

            new_initial = self._generate_initial(mapping)
            if new_initial:
                initials.append(new_initial)

            if isinstance(mapping, celosia_signal.Port):
                ports.append(f'{self._generate_port(mapping)}')

                if mapping.direction == 'i':
                    continue

            if mapping.static:
                if mapping.statements:
                    statement = mapping.statements[0]
                elif mapping.reset_statement is not None:
                    statement = mapping.reset_statement
                else:
                    statement = None

                if statement:
                    assignments.append(self._generate_assignment(mapping, statement))
            else:
                blocks.append(f'{self._generate_block(mapping)}')

            self._generate_signal_features(mapping)

        ports = f'{self.portsep}\n'.join(ports)
        initials = '\n'.join(initials)
        assignments = '\n'.join(assignments)
        blocks = '\n'.join(blocks)

        return ports, initials, assignments, blocks

    def _generate_submodule(self, submodule: celosia_module.Module) -> str:
        return ''

    def _generate_submodule_features(self, submodule: celosia_module.Module):
        return

    def _generate_submodules(self) -> str:
        submodules: list[str] = []

        for submodule in self.module.submodules:
            if submodule.empty:
                continue

            submodules.append(self._generate_submodule(submodule))
            self._generate_submodule_features(submodule)

        return '\n'.join(submodules)

    def _generate_switch(self, mapping: celosia_signal.Signal, statement: celosia_statement.Statement) -> str:
        return ''

    def _generate_statements(self, mapping: celosia_signal.Signal, statements: list[celosia_statement.Statement]) -> str:
        res = []
        for statement in statements:
            new = self._generate_statement(mapping, statement)
            if new:
                res.append(new)
        return '\n'.join(res)

    def _generate_statement(self, mapping: celosia_signal.Signal, statement: celosia_statement.Statement):
        if isinstance(statement, celosia_statement.Assign):
            res = self._generate_assignment(mapping, statement)

        elif isinstance(statement, celosia_statement.Switch):
            res = self._generate_switch(mapping, statement)

        else:
            raise RuntimeError(f"Unknown statement: {statement}")

        return res

    def tabs(self, n: int = 1) -> str:
        if n <= 0:
            return ''

        return ' ' * (self.spaces * n)

    def _parse_parameter(self, parameter: Any):
        if isinstance(parameter, int):
            parameter = self._parse_parameter(_ast.Const(parameter, parameter.bit_length()))
        elif isinstance(parameter, float):
            pass
        elif isinstance(parameter, str):
            parameter = f'"{self._escape_string(parameter)}"'
        else:
            raise ValueError(f"Unknown parameter object detected: {parameter} (type {type(parameter)})")
        return parameter

    def _parse_rhs(self, rhs: Any):
        raise ValueError(f"Unknown RHS object detected: {rhs} (type {type(rhs)})")

    def _parse_attribute(self, key: str, value: Any) -> str:
        return str(self._parse_parameter(value))

    @classmethod
    def _escape_string(cls, string: str) -> str:
        return string

def get_lang_map() -> dict[str, type[HDL]]:
    lang_map: dict[str, type[HDL]] = {}
    for _, name, _ in pkgutil.iter_modules(__path__, __name__ + '.'):
        for HDLType in importlib.import_module(name).__dict__.values():
            if isinstance(HDLType, type) and issubclass(HDLType, HDL):
                lang_map[name.split('.')[-1]] = HDLType
    return lang_map