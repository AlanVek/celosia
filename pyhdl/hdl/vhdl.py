from pyhdl.hdl import HDL
import pyhdl.backend.signal as pyhdl_signal
import pyhdl.backend.module as pyhdl_module
import pyhdl.backend.statement as pyhdl_statement
from textwrap import indent
from amaranth.hdl import ast, ir
from typing import Union

class VHDL(HDL):
    case_sensitive = False
    portsep = ';'
    top_first = False

    protected = [
        'abs',                  'access',         'after',          'alias',          'all',
        'and',                  'architecture',   'array',          'assert',         'attribute',
        'begin',                'block',          'body',           'buffer',         'bus',
        'case',                 'component',      'configuration',  'constant',       'context',
        'cover',                'default',        'disconnect',     'downto',         'else',
        'elsif',                'end',            'entity',         'exit',           'fairness',
        'file',                 'for',            'force',          'function',       'generate',
        'generic',              'group',          'guarded',        'if',             'impure',
        'in',                   'inertial',       'inout',          'is',             'label',
        'library',              'linkage',        'literal',        'loop',           'map',
        'mod',                  'nand',           'new',            'next',           'nor',
        'not',                  'null',           'of',             'on',             'open',
        'or',                   'others',         'out',            'package',        'parameter',
        'port',                 'postponed',      'procedure',      'process',        'property',
        'protected',            'pure',           'range',          'record',         'register',
        'reject',               'release',        'rem',            'report',         'restrict',
        'restrict_guarantee',   'return',         'rol',            'ror',
        'select',               'sequence',       'severity',       'shared',         'signal',
        'sla',                  'sll',            'sra',            'srl',            'strong',
        'subtype',              'then',           'to',             'transport',      'type',
        'unaffected',           'units',          'until',          'use',            'variable',
        'view',                 'vmode',          'vprop',          'vunit',          'wait',
        'when',                 'while',          'with',           'xnor',           'xor',
    ]

    template = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.numeric_std_unsigned.all;

entity {name} is
    port ({ports}
    );
end {name};

architecture rtl of {name} is{components}{types}{initials}
begin{submodules}{blocks}{assignments}
end rtl;
"""

    def __init__(self, spaces: int = 4, blackboxes: list[dict[str, Union[int, str, tuple]]] = None):
        super().__init__(spaces=spaces)
        self._blackboxes = blackboxes
        self._types: dict[pyhdl_signal.Memory, list[tuple[int, str]]] = {}
        self._typenames: list[str] = []
        self._processes: set[str] = set()

        if blackboxes is not None:
            raise NotImplementedError("Blackboxes not supported yet!")

    def sanitize(self, name: str) -> str:
        name = super().sanitize(name).strip()

        replace_map = {
            '\\': '',
            '$': '_esc_',
            '.': '_',
            ':': '_',
            '[': '_',
            ']': '_',
            '(': '_',
            ')': '_',
            '{': '_',
            '}': '_',
            '-': '_',
        }

        for old, new in replace_map.items():
            name = name.replace(old, new)

        while '__' in name:
            name = name.replace('__', '_')

        if name and name[0] == '_':
            name = name[1:]
        if name and name[-1] == '_':
            name = name[:-1]

        while name in self.protected:
            name = 'esc_' + name

        if not name:
            name = self.sanitize('unnamed')

        if name[0].isnumeric():
            name = 'esc_' + name

        # TODO: This is not populated yet
        # while name in self._typenames:
        #     name = 'esc_' + name
        # while name in self._processes:
        #     name = 'esc_' + name

        return name

    def reset(self):
        super().reset()
        self._types.clear()
        self._typenames.clear()
        self._processes.clear()

        self.signal_features['types'] = []
        self.submodule_features['components'] = []

    def _generate_memory_type(self, mapping: pyhdl_signal.Memory):
        new_type = 'std_logic'
        signal = mapping.signal
        depth = len(mapping.init)

        self._types[mapping] = []

        # TODO: Check that type doesn't collide with some name
        for i, size in enumerate([len(signal), depth]):
            next_type = f'type_{signal.name}_{i}'
            self._types[mapping].append((len(self._typenames), f'array (0 to {size - 1}) of {new_type}'))
            self._typenames.append(next_type)
            new_type = next_type

    def _get_memory_type(self, mapping: pyhdl_signal.Signal) -> list[tuple[str, str]]:
        if not isinstance(mapping, pyhdl_signal.Memory):
            return None

        if mapping not in self._types:
            self._generate_memory_type(mapping)

        return self._types[mapping]

    def _get_memory_typename(self, mapping: pyhdl_signal.Signal, idx: int) -> list[tuple[str, str]]:
        entry = self._get_memory_type(mapping)

        if entry is None:
            return None

        return self._typenames[entry[idx][0]]

    def _generate_signal_features(self, mapping: pyhdl_signal.Signal):
        types = self._get_memory_type(mapping)
        if types is None:
            return

        for idx, description in types:
            self.signal_features['types'].append(f'type {self._typenames[idx]} is {description};')

    def _generate_signal_from_string(self, name: str, width: Union[int, ast.Shape], dir: str = None, type=None) -> str:
        # TODO: Check what to do with zero-width signals
        # if len(mapping.signal) <= 0:
        #     raise RuntimeError(f"Zero-width mapping {mapping.signal.name} not allowed")

        if dir is None:
            dir = ''
        else:
            if dir == 'i':
                dir = 'in'
            elif dir == 'o':
                dir = 'out'
            elif dir == 'io':
                dir = 'inout'
            dir = f'{dir} '

        if type is None:
            type = f'std_logic_vector({width - 1} downto 0)'

        return f'{name} : {dir}{type}'

    def _generate_signal(self, mapping, dir=None):
        if isinstance(mapping, pyhdl_signal.Memory):
            if len(mapping.init) <= 0:
                raise RuntimeError(f"Zero-depth memory {mapping.signal.name} not allowed")

            type = self._get_memory_typename(mapping, -1)
        else:
            type = None

        return self._generate_signal_from_string(mapping.signal.name, len(mapping.signal), dir=dir, type=type)

    def _generate_port(self, port: pyhdl_signal.Port):
        ret = self._generate_signal(port, port.direction)

        if port.direction == 'o':
            ret += f' := {self._generate_reset(port.signal.reset, len(port.signal))}'

        return indent(ret, self.tabs())

    def _generate_port_from_string(self, name: str, width: int, dir: str, type=None):
        return self._generate_signal_from_string(name, width, dir=dir, type=type)

    def _generate_reset(self, values, width) -> str:
        if isinstance(values, (int, ast.Const)):
            values = [values]
        elif not isinstance(values, (list, tuple, set)):
            raise RuntimeError(f"Unknown reset value: {values}")

        resets = []

        for value in values:
            if isinstance(value, ast.Const):
                width = max(width, value.width)
                value = value.value

            binary_reset = format(value, f'0{width}b')
            b0 = binary_reset[0]

            if all(b == b0 for b in binary_reset):
                reset = f"(others => '{b0}')"
            else:
                reset = f'"{binary_reset}"'

            resets.append(reset)

        if len(resets) == 1:
            return resets[0]
        elif all(r == reset[0] for r in resets):
            return f"(others => {resets[0]})"
        else:
            return '\n'.join((
                '(',
                indent(',\n'.join(f'{i} => {reset}' for i, reset in enumerate(resets)), self.tabs()),
                ')',
            ))

    def _generate_initial(self, mapping: pyhdl_signal.Signal):
        # pyhdl_signal.Memory ports don't have initials, they're created with the parent signal's pyhdl_signal.Memory
        if isinstance(mapping, pyhdl_signal.MemoryPort):
            return ''

        # Ports don't have initials
        if isinstance(mapping, pyhdl_signal.Port):
            return ''

        if isinstance(mapping, pyhdl_signal.Memory):
            reset = mapping.init
        else:
            reset = mapping.signal.reset

        return f'signal {self._generate_signal(mapping)} := {self._generate_reset(reset, len(mapping.signal))};'

    def _generate_assignment(self, mapping: pyhdl_signal.Signal, statement: pyhdl_statement.Assign):
        start_idx = statement._start_idx
        stop_idx = statement._stop_idx

        if start_idx is None:
            start_idx = 0
        if stop_idx is None:
            stop_idx = len(mapping.signal)

        repr = mapping.signal.name

        size = len(mapping.signal)
        rhs_wrapper = lambda x: x

        if isinstance(mapping, pyhdl_signal.MemoryPort):
            repr = f'{repr}(to_integer({self._parse_rhs(mapping.index)}))'
            rhs_wrapper = lambda x: f'{self._get_memory_typename(mapping.memory, 0)}({x})'

        if start_idx != 0 or stop_idx != len(mapping.signal):
            repr = f'{repr}({stop_idx-1} downto {start_idx})'
            size = min(size, stop_idx - start_idx)

        if mapping.domain is None:
            posfix = ''
        else:
            # FIX: Need to add small delay for simulation
            posfix = 'after 1 fs'

        return f'{repr} <= {rhs_wrapper(self._parse_rhs(statement.rhs, size))} {posfix};'

    def _generate_block(self, mapping: pyhdl_signal.Signal) -> str:
        statements: list[pyhdl_statement.Statement] = []

        if mapping.reset_statement is not None:
            statements.append(mapping.reset_statement)
        statements.extend(mapping.statements)

        if not statements:
            return ''

        if mapping.domain is None:
            sensitivity = ['all']
            triggers = None
        else:
            sensitivity = []
            triggers = []

            domain = mapping.domain

            clk = self._parse_rhs(domain.clk)
            sensitivity.append(clk)

            if domain.clk_edge == 'pos':
                edge = 'rising'
            else:
                edge = 'falling'

            triggers.append(f'{edge}_edge({clk}(0))')

            if domain.async_reset and domain.rst is not None:
                rst = self._parse_rhs(domain.rst)
                sensitivity.append(rst)
                triggers.append(f'rising_edge({rst}(0))')

            triggers = f'({" or ".join(triggers)})'

        tabs = 1 + bool(triggers)
        if triggers:
            header = f'if {triggers} then'
            footer = 'end if;'
        else:
            header = footer = ''

        # TODO: Check that process name doesn't collide with some name
        new_process = f'p_{mapping.signal.name}'
        self._processes.add(new_process)

        return '\n'.join((
            f'{new_process}: process ({", ".join(sensitivity)})',
            'begin',
            *([indent(header, self.tabs())] if header else []),
            indent(self._generate_statements(mapping, statements), self.tabs(tabs)),
            *([indent(footer, self.tabs())] if footer else []),
            'end process;',
            '', # Add new line at the end to separate blocks
        ))

    def _generate_if(self, mapping: pyhdl_signal.Signal, statement: pyhdl_statement.Statement, as_if: list[Union[pyhdl_statement.Switch.If, pyhdl_statement.Switch.Else]]):
        if_opening = 'if'
        else_done = False

        ret = ''
        for i, case in enumerate(as_if):
            if else_done:
                raise RuntimeError("New case after 'else'")

            if isinstance(case, pyhdl_statement.Switch.If):
                opening = if_opening
                if_opening = 'elsif'
                begin = ' then'
            elif isinstance(case, pyhdl_statement.Switch.Else):
                opening = 'else'
                else_done = True
                begin = ''

            if case.test is None:
                test = ''
            else:
                idx = lambda x: x
                if isinstance(case.test, ast.Signal):
                    idx = lambda x: f'{x}(0)'

                test = str(self._parse_rhs(case.test, allow_bool=True))

                # FIX: If '0'/'1' not allowed apparently
                for possible_value in [0, 1]:
                    if test == self._parse_rhs(ast.Const(possible_value, 1)[0], allow_bool=True):
                        test = f"std_logic'('{possible_value}') = std_logic'('1')"
                        idx = lambda x: x
                        break
                test = f' {idx(test)}'

            ret += f'{opening}{test}{begin}'

            if case.statements:
                case_body = self._generate_statements(mapping, case.statements)
            else:
                case_body = ''

            ret += '\n' + indent(case_body, self.tabs())
            if case_body:
                ret += '\n'

            if i >= len(as_if) - 1:
                ret += 'end if;'

        return ret

    def _generate_switch(self, mapping: pyhdl_signal.Signal, statement: pyhdl_statement.Switch):
        if statement.as_if is not None:
            return self._generate_if(mapping, statement, statement.as_if)

        cases = statement.cases.copy()
        cases.setdefault(None, [])

        body = []
        for case, statements in cases.items():
            if case is None:
                case = 'others'
            else:
                case = case.replace('?', '-')
                case = f'"{case}"'

            body.append(f'{self.tabs()}when {case} =>')

            if statements:
                case_body = self._generate_statements(mapping, statements)
            else:
                case_body = '-- empty --;'

            body.append(indent(case_body, self.tabs(2)))

        return '\n'.join((
            f'case? ({self._parse_rhs(statement.test)}) is',
            *body,
            'end case?;',
        ))

    def _generate_submodule(self, submodule: pyhdl_module.Module, ports: dict[str, pyhdl_signal.Port], parameters: dict):
        res = []

        if parameters:
            # TODO: Check generics types
            res.append('\n'.join((
                'generic map (',
                indent(',\n'.join(f'{name} => {value}' for name, value in parameters.items()), self.tabs()),
                ')'
            )))

        res.append('port map (')
        if ports:
            res.append(
                indent(',\n'.join(f'{name} => {self._parse_rhs(value.signal)}' for name, value in ports.items()), self.tabs()),
            )

        res.append(');')

        if isinstance(submodule, pyhdl_module.InstanceModule):
            prefix = ''
        else:
            prefix = 'entity work.'

        return '\n'.join((
            f'{submodule.name}: {prefix}{submodule.type}',
            indent('\n'.join(res), self.tabs()),
        ))

    def _generate_submodule_features(self, submodule: pyhdl_module.Module, ports: dict[str, pyhdl_signal.Port], parameters: dict):
        # TODO: Support blackboxes

        res = []

        # if parameters:
        #     # TODO: Check generics types and parse defaults
        #     res.append('\n'.join((
        #         'generic (',
        #         indent(';\n'.join(f'{name} : {type} := {default}' for name, (type, default) in parameters.items()), self.tabs()),
        #         ');'
        #     )))

        # res.append('port (')
        # if ports:
        #     res.append(
        #         indent(';\n'.join(
        #             f'{self._generate_port_from_string(name, len(value.signal), value.direction)}' for name, value in ports.items()
        #         ), self.tabs()),
        #     )
        # res.append(');')

        # self.submodule_features['components'].extend((
        #     f'component {submodule.type} is',
        #     indent('\n'.join(res), self.tabs()),
        #     '\nend component;',
        #     ''
        # ))

    def _parse_rhs(self, rhs: Union[ast.Value, int, str, pyhdl_signal.MemoryPort], size: int = None, allow_signed: bool = True, allow_bool: bool = False, operation: bool = False):
        if isinstance(rhs, pyhdl_signal.MemoryPort):
            size = len(rhs.signal)

        if size is None:
            size = len(rhs)

        if isinstance(rhs, ast.Const):
            signed = rhs.signed
            value = rhs.value
            width = rhs.width

            if value < 0:
                value += 2**width

            if width % 4:
                value = format(value, f'0{width}b')
                value = f'"{value}"'
            else:
                value = format(value, f'0{width//4}x')
                value = f'x"{value}"'

            if signed:
                fn = 'signed'
            else:
                fn = 'unsigned'

            rhs = f"{fn}(std_logic_vector'({value}))"

            if not operation:
                rhs = f'std_logic_vector({rhs})'

        elif isinstance(rhs, int):
            pass

        elif isinstance(rhs, str):
            rhs = f'"{rhs}"'

        elif isinstance(rhs, ast.Signal):
            signed = allow_signed and rhs.shape().signed
            width = len(rhs)
            rhs = rhs.name

            if width < size:
                if signed:
                    fill = f"{rhs}({width - 1})"
                else:
                    fill = "'0'"
                rhs = f"({width - 1} downto 0 => {rhs}, others => {fill})"

            elif width > size:
                rhs = f'{rhs}({size-1} downto 0)'

            if operation:
                if signed:
                    rhs = f'signed({rhs})'
                else:
                    rhs = f'unsigned({rhs})'

        elif isinstance(rhs, ast.Cat):
            rhs = f"{' & '.join(self._parse_rhs(part) for part in rhs.parts[::-1])}"
        elif isinstance(rhs, ast.Slice):
            if (rhs.start == 0 and rhs.stop >= len(rhs.value)) and not allow_bool:
                rhs = self._parse_rhs(rhs.value)
            else:
                if allow_bool and (rhs.stop == rhs.start + 1):
                    idx = rhs.start
                else:
                    idx = f'{rhs.stop-1} downto {rhs.start}'
                rhs = f"{self._parse_rhs(rhs.value, allow_signed=False)}({idx})"

        elif isinstance(rhs, ast.Operator):
            allow_signed = rhs.operator != 'u'
            operation = len(rhs.operands) > 1 and rhs.operator in (
                '+', '-', '*', '//', '%', '<', '<=', '==', '!=', '>', '>=',
            )
            parsed = list(map(lambda x: self._parse_rhs(x, allow_signed=allow_signed, operation=operation), rhs.operands))
            if len(rhs.operands) == 1:
                p0 = parsed[0]
                if rhs.operator == '+':
                    rhs = p0
                elif rhs.operator == '~':
                    rhs = f'not {p0}'
                elif rhs.operator == '-':
                    rhs = f'std_logic_vector(-signed({p0}))'
                elif rhs.operator == 'b':
                    if allow_bool:
                        rhs = f'{p0} /= {self._parse_rhs(ast.Const(0, len(rhs.operands[0])))}'
                    else:
                        rhs = f'"0" when {p0} = {self._parse_rhs(ast.Const(0, len(rhs.operands[0])))} else "1"'
                elif rhs.operator in ('r|', 'r&', 'r^'):
                    if len(rhs.operands[0]) == 1:
                        rhs = str(p0)
                    else:
                        operator = rhs.operator[-1].replace("|", "or").replace('&', 'and').replace('^', 'xor') + ' '
                        rhs = f"(0=>{operator}{p0}, others=>'0')"

                elif rhs.operator == "u":
                    rhs = p0
                elif rhs.operator == "s":
                    if rhs.operands[0].shape().signed:
                        rhs = p0
                    else:
                        rhs = f'signed({p0})'
                        if not operation:
                            rhs = f'std_logic_vector({rhs})'
                else:
                    raise RuntimeError(f"Unknown operator and operands: {rhs.operator}, {rhs.operands}")

            elif len(rhs.operands) == 2:
                p0, p1 = parsed

                resize = lambda x: x

                if operation:
                    resize = lambda x: f'std_logic_vector({x})'

                if rhs.operator in ('+', '-', '*', '//', '%'):
                    if any(len(operand) <= size for operand in rhs.operands):
                        resize = lambda x: f'std_logic_vector(resize({x}, {size}))'

                if rhs.operator in ('+', '-', '*', '//', '%', '&', '^', '|'):
                    operator = rhs.operator[0].replace("|", "or").replace('&', 'and').replace('^', 'xor')
                    rhs = resize(f'{p0} {operator} {p1}')
                elif rhs.operator in ('<', '<=', '==', '!=', '>', '>='):
                    operator = rhs.operator.replace('==', '=').replace('!=', '/=')
                    rhs = f'{p0} {operator} {p1}'
                    if not allow_bool:
                        rhs = f'"1" when {rhs} else "0"'
                elif rhs.operator in ('<<', '>>'):
                    offset = f'to_integer({self._parse_rhs(rhs.operands[1])})'

                    if rhs.operator == '>>':
                        fn = 'shift_right'
                    else:
                        fn = 'shift_left'
                    rhs = f'{fn}({p0}, {offset})'
                else:
                    raise RuntimeError(f"Unknown operator and operands: {rhs.operator}, {rhs.operands}")
            elif len(rhs.operands) == 3:
                p0, p1, p2 = parsed
                if rhs.operator == "m":
                    if any(len(operand) > size for operand in rhs.operands[1:]):
                        p1 = f'{p1}({size-1} downto 0)'
                        p2 = f'{p2}({size-1} downto 0)'

                    if isinstance(rhs.operands[0], ast.Const):
                        if rhs.operands[0].value == 0:
                            rhs = str(p2)
                        else:
                            rhs = str(p1)
                    else:
                        idx = lambda x: x
                        if not isinstance(rhs.operands[0], ast.Operator):
                            idx = lambda x: f'{x}(0)'
                        rhs = f'{p1} when {idx(p0)} else {p2}'
                else:
                    raise RuntimeError(f"Unknown operator and operands: {rhs.operator}, {rhs.operands}")
            else:
                raise RuntimeError(f"Unknown operator and operands: {rhs.operator}, {rhs.operands}")

        elif isinstance(rhs, ast.Part):
            if rhs.stride != 1:
                raise RuntimeError("Only Parts with stride 1 supported at end stage!")
            rhs = f'{self._parse_rhs(rhs.value)} >> {self._parse_rhs(rhs.offset)}'
        elif isinstance(rhs, pyhdl_signal.MemoryPort):
            index = f'to_integer({self._parse_rhs(rhs.index)}'
            rhs = f'std_logic_vector({self._parse_rhs(rhs.signal)}({index})))'
        else:
            raise ValueError("Unknown RHS object detected: {}".format(rhs))

        return rhs


def convert(
    module: Union[ir.Fragment, ir.Elaboratable],
    name: str = 'top',
    ports: list[ast.Signal] = None,
    platform = None,
    spaces: int = 4,
    blackboxes: list[dict[str, Union[int, str, tuple]]] = None,
):
    return VHDL(
        spaces = spaces,
        blackboxes = blackboxes,
    ).convert(
        module,
        name = name,
        ports = ports,
        platform = platform,
    )