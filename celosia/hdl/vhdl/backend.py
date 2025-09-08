from celosia.hdl.backend import Module as BaseModule
from typing import Any, Union
from amaranth.back import rtlil

class VHDLModule(BaseModule):

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._types: list[str] = []

        self._curr_line_manager = []

    @staticmethod
    def _const(value: Any):
        if isinstance(value, str):
            value = value.replace('"', '""')
            return f'"{value}"'
        return super()._const(value)

    @classmethod
    def sanitize(cls, name: str) -> str:
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

        while name in cls.protected:
            name = 'esc_' + name

        if not name:
            name = cls.sanitize('unnamed')

        if name[0].isnumeric():
            name = 'esc_' + name

        # TODO: This is not populated yet
        # while name in cls._typenames:
        #     name = 'esc_' + name
        # while name in cls._processes:
        #     name = 'esc_' + name

        return name

    @staticmethod
    def _const_repr(width, value):
        return str(value)   # TODO

        if isinstance(value, str):

            if '-' in value or width % 4:
                format = 'b'
            else:
                format = 'h'
                value = hex(int(value, 2))[2:]



            rhs = f"{self._sign_fn(signed)}'({base}\"{format(value, f'0{width}{base}')}\")"


            if '-' in value:
                format = 'b'
            else:
                format = 'h'
                value = hex(int(value, 2))[2:]

        elif isinstance(value, int):
            format = 'h'
            if value < 0:
                value += 2**int(width)

            value = hex(value)[2:]

        return f"{width}'{format}{value}"

    @classmethod
    def _concat(cls, parts) -> str:
        return f'{{ {" & ".join(parts[::-1])} }}'

    def _emit_assignment_lhs_rhs(self, lhs: str, rhs: str, parse=True):
        if parse:
            lhs = self._represent(lhs)
            rhs = self._represent(rhs)

        self._line(f'{lhs} <= {rhs};')

    def _emit_assignment(self, assignment: rtlil.Assignment):
        self._emit_assignment_lhs_rhs(assignment.lhs, assignment.rhs)

    def _emit_process_assignment(self, assignment: rtlil.Assignment, comb = True):
        self._emit_assignment_lhs_rhs(assignment.lhs, assignment.rhs)

    def _emit_operator_assignment(self, assignment: rtlil.Assignment, comb = True):
        self._emit_assignment_lhs_rhs(assignment.lhs, assignment.rhs)

    def _emit_submodule(self, submodule: rtlil.Cell):
        super()._emit_submodule(submodule)
        with self._line.indent():
            # TODO: Check instance for entity.work
            self._line(f'{submodule.name}: {submodule.kind}')
            if submodule.parameters:
                self._line('generic map (')
                with self._line.indent():
                    for i, (name, value) in enumerate(submodule.parameters.items()):
                        sep = "," if i < len(submodule.parameters) - 1 else ""
                        self._line(f'{name} => {self._const(value)}{sep}')
                self._line(')')

            if submodule.ports:
                self._line('port map (')
                with self._line.indent():
                    for i, (name, value) in enumerate(submodule.ports.items()):
                        sep = "," if i < len(submodule.ports) - 1 else ""
                        self._line(f'{name} => {self._represent(value)}{sep}')
                self._line(');')

    def _emit_signal(self, signal: rtlil.Wire):
        if signal.port_kind is not None:
            return
        with self._line.indent():
            init = self._get_initial(signal)
            depth = len(init) if isinstance(init, list) else 0

            # TODO: Attributes
            # for key, attr in signal.attributes.items():
            #     self._line(f'(* {key} = {self._const(attr)} *)')

            if depth:
                pass
                # TODO: Memories
                # self._line(f'{dir}{type} {width}{signal.name} [{depth-1}:0];')
                # self._line('initial begin')
                # with self._line.indent():
                #     for i, value in enumerate(init):
                #         self._emit_assignment_lhs_rhs(self._slice_repr(signal.name, i, i), value, parse=False)
                # self._line('end')
            else:
                reset = "(others => '0')" if init is None else init
                self._line(f'signal {signal.name}: std_logic_vector({signal.width - 1} downto 0) := {reset};')

    def _emit_module_definition(self):
        for line in [
            "library ieee;",
            "use ieee.std_logic_1164.all;",
            "use ieee.numeric_std.all;",
            "use ieee.numeric_std_unsigned.all;",
        ]:
            self._line(line)

        kind_map = {
            'input': 'in',
            'output': 'out',
            'inout': 'inout',
        }

        self._line(f'entity {self.name} is')
        with self._line.indent():
            self._line('port (')
            with self._line.indent():
                for i, port in enumerate(self._emitted_ports):
                    reset = " := (others => '0')" if port.port_kind in ('i', 'io') else ''
                    sep = '' if i >= len(self._emitted_ports) - 1 else ';'
                    self._line(f'{port.name}: {kind_map[port.port_kind]} std_logic_vector({port.width-1} downto 0){reset}{sep}')
            self._line(');')
        self._line(f'end {self.name};')

        self._line(f'architecture rtl of {self.name} is')

    def _emit_post_callback(self):
        self._line('begin')

    def _emit_process_start(self, clock = None, polarity: bool = True, arst: str = None, arst_polarity = False) -> str:
        ret = super()._emit_process_start(clock, polarity, arst, arst_polarity)

        if clock is None:
            sensitivity = ['all']
        else:
            clock = self._represent(clock)
            sensitivity = [clock]
            if arst is not None:
                arst = self._represent(arst)
                sensitivity.append(arst)

        self._line(f'{ret}: process ({", ".join(sensitivity)})')
        self._line('begin')

        if clock is not None:
            trigger = f'{"rising" if polarity else "falling"}_edge {clock}'
            if arst is not None:
                trigger += f' or {"rising" if arst_polarity else "falling"}_edge {arst}'

            # TODO: Nasty
            self._curr_line_manager.append(self._line.indent())
            self._curr_line_manager[-1].__enter__()
            self._line(f'if ({trigger}) then')

        return ret

    def _emit_process_end(self, p_id: str, comb=True):
        if not comb:
            assert len(self._curr_line_manager)
            self._line('end if;')
            # TODO: Nasty x2
            self._curr_line_manager.pop().__exit__(None, None, None)
        self._line('end process;')

    @classmethod
    def _slice_repr(cls, name: str, start: int, stop: int=None) -> str:
        if stop is None or stop == start:
            idx = start
        else:
            idx = f'{stop} downto {start}'
        return f'{name}({idx})'

    def _emit_switch_start(self, sel: str):
        self._line(f'case? ({sel}) is')

    def _emit_switch_end(self):
        self._line('end case?')

    def _case_patterns(self, pattern: tuple[str, ...]) -> str:
        return f"{' | '.join(pattern)}"

    def _case_default(self) -> str:
        return 'others'

    def _emit_case_start(self, pattern: str):
        self._line(f'when {pattern} =>')

    def _emit_case_end(self):
        pass

    def _emit_module_end(self):
        self._line('end rtl;')

    def _emit_if_start(self, sel: str):
        self._line(f'if ({sel}) then')

    def _emit_elseif_start(self, sel: str):
        self._line(f'elsif ({sel}) then')

    def _emit_else(self):
        self._line(f'else')

    def _emit_if_end(self):
        self._line('end if')

    def _signed(self, value) -> str:
        # TODO: This doesn't work for constants
        return f'signed({value})'

    def _operator_repr(self, operator: rtlil.Cell) -> str:
        # TODO: Any issues with constant unary?
        UNARY_OPERATORS = {
            "$neg": "-",
            "$not": "not",
            "$reduce_bool": "or",    # TODO: Check
            "$reduce_or": "or",
            "$reduce_and": "and",
            "$reduce_xor": "xor",
        }
        BINARY_OPERATORS = {
            "$add": '+',
            "$sub": '-',
            "$mul": '*',
            "$divfloor": '/',
            "$modfloor": '%',
            "$shl": '<<', # TODO: missing
            "$shr": '>>', # TODO: missing
            "$sshr": '>>', # TODO: Check sign? # TODO: missing
            "$and": 'and',
            "$or": 'or',
            "$xor": 'xor',
            "$eq": '=',
            "$ne": '!=',
            "$lt": '<',
            "$gt": '>',
            "$le": '<=',
            "$ge": '>=',
            "$shift": '>>', # TODO: missing
        }

        rhs = None

        if operator.kind in UNARY_OPERATORS:
            operands = []
            for port in ('A',):
                operand = self._represent(operator.ports[port])
                if operator.parameters[f'{port}_SIGNED']:
                    operand = self._signed(operand)
                operands.append(operand)

            rhs = f'{UNARY_OPERATORS[operator.kind]} {operands[0]}'

        elif operator.kind in BINARY_OPERATORS:
            operands = []
            for port in ('A', 'B'):
                operand = self._represent(operator.ports[port])
                if operator.parameters[f'{port}_SIGNED']:
                    operand = self._signed(operand)
                operands.append(operand)

            rhs = f' {BINARY_OPERATORS[operator.kind]} '.join(operands)

        elif operator.kind == '$mux':
            operands = []
            for port in ('S', 'B', 'A'):
                operands.append(self._represent(operator.ports[port]))
            rhs = f'{operands[0]} ? {operands[1]} : {operands[2]}'

        if rhs is None:
            raise RuntimeError(f"Unknown operator: {operator.kind}")

        return rhs