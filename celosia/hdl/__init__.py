from textwrap import indent
import celosia.backend.signal as celosia_signal
import celosia.backend.module as celosia_module
import celosia.backend.statement as celosia_statement
from amaranth.hdl import ast, ir
from typing import Union

# TODO: If we find multiple signals with same statements, maybe we can merge them into one!

class HDL:
    case_sensitive = False
    portsep = ','
    top_first = True

    template = """{name}
{ports}{initials}{submodules}{blocks}{assignments}
"""

    def __init__(self, spaces: int = 4):
        self.spaces = spaces
        self.signal_features: dict[str, list] = {}
        self.submodule_features: dict[str, list] = {}

        self.module: celosia_module.Module = None
        self.invalid_names: set[str] = set()

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
            if self._change_case(name) in invalid:
                name = name[:curr_idx+1]
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

    def convert(self, fragment: Union[ir.Fragment, ir.Elaboratable], name: str = 'top', ports: list[ast.Signal] = None, platform=None):
        m = celosia_module.Module(name, fragment)
        m.prepare(ports, platform)

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

        converted = self.template.format(**formats)
        res = []

        if self.top_first:
            res.append(converted)

        for submodule in self.module.submodules:
            if isinstance(submodule, celosia_module.InstanceModule):
                continue
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