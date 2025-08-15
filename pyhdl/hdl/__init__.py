from textwrap import indent
import pyhdl.backend.signal as pyhdl_signal
import pyhdl.backend.module as pyhdl_module
import pyhdl.backend.statement as pyhdl_statement
from amaranth.hdl import ast, ir

# TODO: If we find multiple signals with same statements, maybe we can merge them into one!

class HDL:
    case_sensitive = False
    portsep = ','

    template = """{name}
{ports}{initials}{submodules}{blocks}{assignments}
"""

    def __init__(self, spaces: int = 4):
        self.spaces = spaces
        self.signal_features: dict[str, list] = {}
        self.submodule_features: dict[str, list] = {}

        self.module: pyhdl_module.Module = None

    @classmethod
    def sanitize(self, name: str) -> str:
        return name

    def convert(self, fragment: ir.Fragment, name: str = 'top', ports: list[ast.Signal] = None, platform=None):
        m = pyhdl_module.Module(name, fragment, hdl=self)
        m.prepare(ports, platform)

        return self._convert_module(m)

    def reset(self):
        self.signal_features.clear()
        self.submodule_features.clear()

    def _convert_module(self, module: pyhdl_module.Module) -> str:
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

        res = [self.template.format(**formats)]
        for submodule, _ in self.module.submodules:
            if isinstance(submodule, pyhdl_module.InstanceModule):
                continue
            res.append(self._convert_module(submodule))

        return '\n'.join(res)

    def _generate_port(self, mapping: pyhdl_signal.Signal) -> str:
        return ''

    def _generate_initial(self, mapping: pyhdl_signal.Signal) -> str:
        return ''

    def _generate_assignment(self, mapping: pyhdl_signal.Signal, statement: pyhdl_statement.Statement) -> str:
        return ''

    def _generate_block(self, mapping: pyhdl_signal.Signal)  -> str:
        return ''

    def _generate_signal_features(self, mapping: pyhdl_signal.Signal):
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

            if isinstance(mapping, pyhdl_signal.Port):
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

    def _generate_submodule(self, submodule: pyhdl_module.Module, ports: dict[str, pyhdl_signal.Port], parameters: dict) -> str:
        return ''

    def _generate_submodule_features(self, submodule: pyhdl_module.Module, ports: dict[str, pyhdl_signal.Port], parameters: dict):
        return

    def _generate_submodules(self) -> str:
        submodules: list[str] = []

        for submodule, ports in self.module.submodules:
            params = {}
            if not isinstance(submodule, pyhdl_module.InstanceModule):
                if ports is not None:
                    raise RuntimeError(f"Found invalid submodule configuration for submodule {submodule.name} of module {self.module.name}")

                if submodule.empty:
                    continue

                ports = {}
                for port in submodule.ports:
                    if not len(port.signal):
                        continue
                    # if port.signal not in module.signals:
                    #     raise RuntimeError(f"Found port {port.signal.name} of submodule {name} which is not a signal of {module.name}")
                    ports[port.signal.name] = port

            else:
                if ports is None:
                    raise RuntimeError(f"Found invalid submodule configuration for submodule {submodule.name} of module {self.module.name}")

                params.update(submodule.parameters)

            submodules.append(self._generate_submodule(submodule, ports, params))
            self._generate_submodule_features(submodule, ports, params)

        return '\n'.join(submodules)

    def _generate_switch(self, mapping: pyhdl_signal.Signal, statement: pyhdl_statement.Statement) -> str:
        return ''

    def _generate_statements(self, mapping: pyhdl_signal.Signal, statements: list[pyhdl_statement.Statement]) -> str:
        res = []
        for statement in statements:
            new = self._generate_statement(mapping, statement)
            if new:
                res.append(new)
        return '\n'.join(res)

    def _generate_statement(self, mapping: pyhdl_signal.Signal, statement: pyhdl_statement.Statement):
        if isinstance(statement, pyhdl_statement.Assign):
            res = self._generate_assignment(mapping, statement)

        elif isinstance(statement, pyhdl_statement.Switch):
            res = self._generate_switch(mapping, statement)

        else:
            raise RuntimeError(f"Unknown statement: {statement}")

        return res

    def tabs(self, n: int = 1) -> str:
        if n <= 0:
            return ''

        return ' ' * (self.spaces * n)