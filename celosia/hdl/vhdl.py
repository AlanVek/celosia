from celosia.hdl import HDL
import celosia.backend.signal as celosia_signal
import celosia.backend.module as celosia_module
import celosia.backend.statement as celosia_statement
from textwrap import indent
from amaranth.hdl import ast, ir
from typing import Union, Any

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
{tabs}port ({ports}
{tabs});
end {name};

architecture rtl of {name} is{components}{types}{initials}
begin{submodules}{blocks}{assignments}
end rtl;
"""

    extension = 'vhd'
    open_comment = '-- '

    def __init__(self, spaces: int = 4, blackboxes: list[dict[str, Union[int, str, tuple]]] = None):
        super().__init__(spaces=spaces)
        self._blackboxes = blackboxes
        self._types: dict[celosia_signal.Memory, list[tuple[int, str]]] = {}
        self._typenames: list[str] = []
        self._processes: set[str] = set()
        self._attrs: dict[str, str] = {}

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
        self._attrs.clear()

        self.signal_features['types'] = []
        self.submodule_features['components'] = []

    def _generate_memory_type(self, mapping: celosia_signal.Memory):
        new_type = 'std_logic'
        signal = mapping.signal
        depth = len(mapping.init)

        self._types[mapping] = []

        # TODO: Check that type doesn't collide with some name
        for i, size in enumerate([len(signal), depth]):
            next_type = f'type_{mapping.name}_{i}'
            self._types[mapping].append((len(self._typenames), f'array (0 to {size - 1}) of {new_type}'))
            self._typenames.append(next_type)
            new_type = next_type

    def _get_memory_type(self, mapping: celosia_signal.Signal) -> list[tuple[str, str]]:
        if not isinstance(mapping, celosia_signal.Memory):
            return None

        if mapping not in self._types:
            self._generate_memory_type(mapping)

        return self._types[mapping]

    def _get_memory_typename(self, mapping: celosia_signal.Signal, idx: int) -> list[tuple[str, str]]:
        entry = self._get_memory_type(mapping)

        if entry is None:
            return None

        return self._typenames[entry[idx][0]]

    def _generate_signal_features(self, mapping: celosia_signal.Signal):
        types = self._get_memory_type(mapping)
        if types is None:
            return

        for idx, description in types:
            self.signal_features['types'].append(f'type {self._typenames[idx]} is {description};')

    def _generate_signal_from_string(self, name: str, width: Union[int, ast.Shape], dir: str = None, type=None) -> str:
        # TODO: Check what to do with zero-width signals
        # if len(mapping.signal) <= 0:
        #     raise RuntimeError(f"Zero-width mapping {mapping.name} not allowed")

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
        if isinstance(mapping, celosia_signal.Memory):
            if len(mapping.init) <= 0:
                raise RuntimeError(f"Zero-depth memory {mapping.name} not allowed")

            type = self._get_memory_typename(mapping, -1)
        else:
            type = None

        return self._generate_signal_from_string(mapping.name, len(mapping.signal), dir=dir, type=type)

    def _generate_port(self, port: celosia_signal.Port):
        ret = self._generate_signal(port, port.direction)

        if port.direction == 'o':
            ret += f' := {self._generate_reset(port.signal.reset, len(port.signal))}'

        return indent(ret, self.tabs())

    def _generate_port_from_string(self, name: str, width: int, dir: str, type=None):
        return self._generate_signal_from_string(name, width, dir=dir, type=type)

    def _generate_reset(self, values, width) -> str:

        depth = None
        if isinstance(values, (int, ast.Const)):
            values = [values]
            depth = 0
        elif not isinstance(values, (list, tuple, set)):
            raise RuntimeError(f"Unknown reset value: {values}")

        if depth is None:
            depth = len(values)

        resets = []

        for value in values:
            if isinstance(value, ast.Const):
                width = max(width, value.width)
                value = value.value

            if value < 0:
                value += 2**width

            binary_reset = format(value, f'0{width}b')
            b0 = binary_reset[0]

            if all(b == b0 for b in binary_reset):
                reset = f"(others => '{b0}')"
            else:
                reset = f'"{binary_reset}"'

            resets.append(reset)

        if depth == 0:
            return resets[0]
        elif all(r == reset[0] for r in resets):
            return f"(others => {resets[0]})"
        else:
            return '\n'.join((
                '(',
                indent(',\n'.join(f'{i} => {reset}' for i, reset in enumerate(resets)), self.tabs()),
                ')',
            ))

    def _parse_attribute(self, key: str, value: Any) -> tuple[str, str, bool]:
        if isinstance(value, bool):
            value = str(value).lower()
            type = 'boolean'

        elif isinstance(value, (int, ast.Const)):
            if isinstance(value, ast.Const):
                value = value.value

            if abs(value) < 2**32:
                type = 'integer'
            else:
                type = 'string'
                value = f'"{value}"'

        else:
            type = 'string'
            if isinstance(value, str):
                value = value.replace('"', '""')
            value = f'"{value}"'

        prev_type = self._attrs.get(key, None)
        if prev_type is None:
            declare = True
            self._attrs[key] = type
        else:
            declare = False
            if prev_type != type:
               raise RuntimeError(f"Unable to generate module '{self.module.name}': attribute '{key}' needs type '{type}' but has already been declared with type '{prev_type}'")

        return type, value, declare

    def _generate_initial(self, mapping: celosia_signal.Signal):
        # celosia_signal.Memory ports don't have initials, they're created with the parent signal's celosia_signal.Memory
        if isinstance(mapping, celosia_signal.MemoryPort):
            return ''

        # Ports don't have initials
        if isinstance(mapping, celosia_signal.Port):
            return ''

        if isinstance(mapping, celosia_signal.Memory):
            reset = mapping.init
        else:
            reset = mapping.signal.reset

        res = f'signal {self._generate_signal(mapping)} := {self._generate_reset(reset, len(mapping.signal))};'

        for key, value in mapping.attrs.items():
            type, value, declare = self._parse_attribute(key, value)
            if declare:
                res += f'\nattribute {key} : {type};'
            res += f'\nattribute {key} of {mapping.name} : signal is {value};'

        return res

    def _generate_assignment(self, mapping: celosia_signal.Signal, statement: celosia_statement.Assign):
        start_idx = statement._start_idx
        stop_idx = statement._stop_idx

        if start_idx is None:
            start_idx = 0
        if stop_idx is None:
            stop_idx = len(mapping.signal)

        repr = mapping.name

        size = len(mapping.signal)
        rhs_wrapper = lambda x: x

        if isinstance(mapping, celosia_signal.MemoryPort):
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

    def _generate_block(self, mapping: celosia_signal.Signal) -> str:
        statements: list[celosia_statement.Statement] = []

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
        new_process = f'p_{mapping.name}'
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

    def _get_if_condition(self, condition: ast.Value) -> str:

        if len(condition) != 1:
            raise RuntimeError(f"Invalid condition, must have width 1: {condition}")

        if isinstance(condition, ast.Slice):
            condition = celosia_module.MemoryModule._slice_check_const(condition.value, condition.start, condition.stop)

        if isinstance(condition, ast.Const):
            # FIX: If '0'/'1' not allowed apparently
            ret = 'false' if condition.value == 0 else 'true'

        else:
            if isinstance(condition, ast.Signal):
                condition = condition[0]
            ret = str(self._parse_rhs(condition, allow_signed=False, force_bool=True))

        return ret

    def _generate_if(self, mapping: celosia_signal.Signal, statement: celosia_statement.Statement, as_if: list[Union[celosia_statement.Switch.If, celosia_statement.Switch.Else]]):
        if_opening = 'if'
        else_done = False

        ret = ''
        for i, case in enumerate(as_if):
            if else_done:
                raise RuntimeError("New case after 'else'")

            if isinstance(case, celosia_statement.Switch.If):
                opening = if_opening
                if_opening = 'elsif'
                begin = ' then'
            elif isinstance(case, celosia_statement.Switch.Else):
                opening = 'else'
                else_done = True
                begin = ''

            test = '' if case.test is None else f' {self._get_if_condition(case.test)}'
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

    def _generate_switch(self, mapping: celosia_signal.Signal, statement: celosia_statement.Switch):
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

    def _generate_submodule(self, submodule: celosia_module.Module):
        res = []

        if submodule.parameters:
            # TODO: Check generics types
            res.append('\n'.join((
                'generic map (',
                indent(',\n'.join(f'{name} => {value}' for name, value in submodule.parameters.items()), self.tabs()),
                ')'
            )))

        res.append('port map (')
        if submodule.ports:
            res.append(
                indent(',\n'.join(f'{port.name} => {self._parse_rhs(port.signal)}' for port in submodule.ports), self.tabs()),
            )

        res.append(');')

        if isinstance(submodule, celosia_module.InstanceModule):
            prefix = ''
        else:
            prefix = 'entity work.'

        return '\n'.join((
            f'{submodule.name}: {prefix}{submodule.type}',
            indent('\n'.join(res), self.tabs()),
        ))

    def _generate_submodule_features(self, submodule: celosia_module.Module):
        # TODO: Support blackboxes

        res = []

        # if submodule.parameters:
        #     # TODO: Check generics types and parse defaults
        #     res.append('\n'.join((
        #         'generic (',
        #         indent(';\n'.join(f'{name} : {type} := {default}' for name, (type, default) in submodule.parameters.items()), self.tabs()),
        #         ');'
        #     )))

        # res.append('port (')
        # if submodule.ports:
        #     res.append(
        #         indent(';\n'.join(
        #             f'{self._generate_port_from_string(port.name, len(port.signal), port.direction)}' for port in submodule.ports
        #         ), self.tabs()),
        #     )
        # res.append(');')

        # self.submodule_features['components'].extend((
        #     f'component {submodule.type} is',
        #     indent('\n'.join(res), self.tabs()),
        #     '\nend component;',
        #     ''
        # ))

    def _parse_rhs(
        self,
        rhs: Union[ast.Value, int, str, celosia_signal.MemoryPort],
        size: int = None,
        allow_signed: bool = True,
        force_bool: bool = False,
        operation: bool = False,
    ) -> Union[str, int]:
        if isinstance(rhs, celosia_signal.MemoryPort):
            size = len(rhs.signal)

        if size is None:
            size = len(rhs)

        if isinstance(rhs, ast.Const):
            signed = rhs.signed
            value = rhs.value
            width = max(size, rhs.width)

            if value < 0:
                value += 2**rhs.width

            if width % 4:
                base = 'b'
            else:
                base = 'x'
                width //= 4

            rhs = f"{self._sign_fn(signed)}'({base}\"{format(value, f'0{width}{base}')}\")"

            if not operation:
                rhs = f'std_logic_vector({rhs})'

        elif isinstance(rhs, int):
            pass

        elif isinstance(rhs, str):
            rhs = rhs.replace('"', '""')
            rhs = f'"{rhs}"'

        elif isinstance(rhs, ast.Signal):
            signed = allow_signed and rhs.shape().signed
            width = len(rhs)
            rhs = self.module.signals.get(rhs).name

            if width < size:
                if signed:
                    fill = f"{rhs}({width - 1})"
                else:
                    fill = "'0'"
                rhs = f"({width - 1} downto 0 => {rhs}, others => {fill})"

            elif width > size:
                rhs = f'{rhs}({size-1} downto 0)'

            if operation:
                rhs = f'{self._sign_fn(signed)}({rhs})'

        elif isinstance(rhs, ast.Cat):
            rhs = f"{' & '.join(self._parse_rhs(part) for part in rhs.parts[::-1])}"

        elif isinstance(rhs, ast.Slice):
            if (rhs.start == 0 and rhs.stop >= len(rhs.value)) and not force_bool:
                rhs = self._parse_rhs(rhs.value)
            else:
                if force_bool and (rhs.stop == rhs.start + 1):
                    idx = rhs.start
                else:
                    idx = f'{rhs.stop-1} downto {rhs.start}'
                rhs = f"{self._parse_rhs(rhs.value, allow_signed=False)}({idx})"

        elif isinstance(rhs, ast.Operator):
            rhs = self._parse_op(rhs, size=size, force_bool=force_bool, operation=operation)

        elif isinstance(rhs, ast.Part):
            if rhs.stride != 1:
                raise RuntimeError("Only Parts with stride 1 supported at end stage!")
            rhs = f'{self._parse_rhs(rhs.value)} >> {self._parse_rhs(rhs.offset)}'
        elif isinstance(rhs, celosia_signal.MemoryPort):
            index = f'to_integer({self._parse_rhs(rhs.index)}'
            rhs = f'std_logic_vector({self._parse_rhs(rhs.signal)}({index})))'
        else:
            raise ValueError(f"Unknown RHS object detected: {rhs}")

        return rhs

    @staticmethod
    def _operator_remap(operator: str) -> str:
        remap = {
            '|': 'or', '&': 'and', '^': 'xor', '%': 'rem', '//': '/',
            '==': '=', '!=': '/=',
        }
        if operator.startswith('r'):
            operator = operator[1:]

        for old, new in remap.items():
            if operator == old:
                return new

        return operator

    @staticmethod
    def _sign_fn(signed: bool) -> str:
        return 'signed' if signed else 'unsigned'

    def _parse_op(self, rhs: ast.Operator, **kwargs) -> Union[int, str]:
        if len(rhs.operands) == 1:
            fn = self._parse_unary_op
        elif len(rhs.operands) == 2:
            fn = self._parse_binary_op
        elif len(rhs.operands) == 3:
            fn = self._parse_ternary_op
        else:
            raise RuntimeError(f"Invalid number of operands: {len(rhs.operands)}")

        return fn(rhs, **kwargs)

    def _parse_unary_op(self, rhs: ast.Operator, size: int, force_bool: bool = False, operation: bool = False) -> Union[int, str]:
        allow_signed = rhs.operator not in ('u', 's')
        parsed = tuple(self._parse_rhs(op, allow_signed=allow_signed) for op in rhs.operands)
        p0, = parsed

        if rhs.operator == '+':
            rhs = p0

        elif rhs.operator == '~':
            rhs = f'not {p0}'

        elif rhs.operator == '-':
            new_rhs = f'-{self._sign_fn(True)}({p0})'

            # FIX: -signal has +1 bit for Amaranth, but not for VHDL
            if len(rhs.operands[0]) < size:
                new_rhs = f'resize({new_rhs}, {size})'
            rhs = f'std_logic_vector({new_rhs})'

        elif rhs.operator == 'b':
            if force_bool:
                rhs = f'or {p0}'
            else:
                rhs = f'"0" when {p0} = {self._parse_rhs(ast.Const(0, len(rhs.operands[0])))} else "1"'

        elif rhs.operator in ('r|', 'r&', 'r^'):
            if len(rhs.operands[0]) == 1:
                rhs = str(p0)
            else:
                rhs = f"(0=>{self._operator_remap(rhs.operator)} {p0}, others=>'0')"

        elif rhs.operator in ('u', 's'):
            rhs = f'{self._sign_fn(rhs.operator == "s")}({p0})'
            if not operation:
                rhs = f'std_logic_vector({rhs})'
        else:
            raise RuntimeError(f"Unknown operator and operands: {rhs.operator}, {rhs.operands}")

        return rhs

    def _parse_binary_op(self, rhs: ast.Operator, size: int, force_bool: bool = False, operation: bool = False) -> Union[int, str]:
        as_op = rhs.operator not in ('<<', '>>', '&', '^', '|')
        parsed = tuple(self._parse_rhs(op, operation=as_op) for op in rhs.operands)
        p0, p1 = parsed

        if rhs.operator in ('+', '-', '*', '//', '%', '&', '^', '|'):
            # Fix: Multiplication increases VHDL size, others don't
            if rhs.operator != '*' and any(len(operand) < size for operand in rhs.operands):
                p0, p1 = (f'resize({p}, {size})' for p in parsed)

            rhs = f'{p0} {self._operator_remap(rhs.operator)} {p1}'
            if as_op:
                rhs = f'std_logic_vector({rhs})'

        elif rhs.operator in ('<', '<=', '==', '!=', '>', '>='):
            rhs = f'{p0} {self._operator_remap(rhs.operator)} {p1}'
            if not force_bool:
                rhs = f'"1" when {rhs} else "0"'

        elif rhs.operator in ('<<', '>>'):
            offset = f'to_integer({p1})'

            if len(rhs.operands[0]) < size:
                p0 = f'resize({p0}, {size})'

            fn = f'shift_{"right" if rhs.operator == ">>" else "left"}'
            rhs = f'{fn}({p0}, {offset})'

        else:
            raise RuntimeError(f"Unknown operator and operands: {rhs.operator}, {rhs.operands}")

        return rhs

    def _parse_ternary_op(self, rhs: ast.Operator, size: int, force_bool: bool = False, operation: bool = False) -> Union[int, str]:
        p0 = self._get_if_condition(rhs.operands[0])
        parsed = tuple(self._parse_rhs(op) for op in rhs.operands[1:])
        p1, p2 = parsed

        if rhs.operator == "m":
            if any(len(operand) > size for operand in rhs.operands[1:]):
                p1 = f'{p1}({size-1} downto 0)'
                p2 = f'{p2}({size-1} downto 0)'

            rhs = f'{p1} when {p0} else {p2}'
        else:
            raise RuntimeError(f"Unknown operator and operands: {rhs.operator}, {rhs.operands}")

        return rhs

def convert(
    module: Union[ir.Fragment, ir.Elaboratable],
    name: str = 'top',
    ports: list[ast.Signal] = None,
    platform = None,
    spaces: int = 4,
    fragment_prepare: bool = True,
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
        fragment_prepare = fragment_prepare,
    )