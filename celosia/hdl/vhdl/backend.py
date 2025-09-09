from celosia.hdl.backend import Module as BaseModule
import celosia.hdl.wire as celosia_wire
from typing import Any, Union
from amaranth.back import rtlil

class VHDLModule(BaseModule):
    submodules_first = True
    case_sensitive = False

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
    def _const_repr(width, value, init=False):
        if isinstance(value, int):
            if value < 0:
                value += 2**width

            if init:
                bin_repr = bin(value)[2:]
                if all(b == bin_repr[0] for b in bin_repr):
                    return f"(others => '{bin_repr[0]}')"

            if width % 4:
                fmt = 'b'
            else:
                fmt = 'x'
                width //= 4

            return f'{fmt}"{format(value, f"0{width}{fmt}")}"'
        return str(value)   # TODO

    @classmethod
    def _concat(cls, parts) -> str:
        return f'( {" & ".join(parts[::-1])} )'

    def _emit_assignment_lhs_rhs(self, lhs: Any, rhs: Any, parse=True, need_cast=False, posfix=None):
        need_fill = False
        if parse:
            lhs_parsed = self._convert_signals(lhs)
            lhs = self._represent(lhs_parsed)
            rhs_parsed = self._convert_signals(rhs)

            need_fill = lhs_parsed.width != rhs_parsed.width

            # Fix: std_logic != std_logic_vector
            need_fill |= isinstance(rhs_parsed, celosia_wire.Slice) and rhs_parsed.width == 1

            rhs = self._represent(rhs_parsed)
            if need_fill:
                rhs = f"({rhs_parsed.width-1} downto 0 => {rhs}, others => '0')"

        if need_cast and not need_fill:
            rhs = f"std_logic_vector({rhs})"

        posfix = '' if posfix is None else f' {posfix}'
        self._line(f'{lhs} <= {rhs}{posfix};')

    def _emit_assignment(self, assignment: rtlil.Assignment):
        self._emit_assignment_lhs_rhs(assignment.lhs, assignment.rhs)

    def _emit_process_assignment(self, assignment: rtlil.Assignment, comb = True):
        self._emit_assignment_lhs_rhs(assignment.lhs, assignment.rhs, posfix=None if comb else 'after 1 fs')

    def _emit_operator_assignment(self, assignment: rtlil.Assignment, comb = True):
        need_cast = assignment.rhs.kind not in {
            '$mux',
            "$eq",
            "$ne",
            "$lt",
            "$gt",
            "$le",
            "$ge",
            "$reduce_bool",
            "$reduce_or",
            "$reduce_and",
            "$reduce_xor",
        }
        self._emit_assignment_lhs_rhs(assignment.lhs, assignment.rhs, need_cast=need_cast)

    def _emit_submodule(self, submodule: rtlil.Cell):
        super()._emit_submodule(submodule)
        with self._line.indent():
            # TODO: Check instance for entity.work
            self._line(f'{submodule.name}: entity work.{submodule.kind}')
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
                    init = self._get_initial(port)
                    reset = "(others => '0')" if init is None else init
                    sep = '' if i >= len(self._emitted_ports) - 1 else ';'
                    self._line(f'{port.name}: {kind_map[port.port_kind]} std_logic_vector({port.width-1} downto 0) := {reset}{sep}')
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
            trigger = f'{"rising" if polarity else "falling"}_edge({self._represent(clock, boolean=True)})'
            if arst is not None:
                trigger += f' or {"rising" if arst_polarity else "falling"}_edge({self._represent(arst, boolean=True)})'

            # TODO: Nasty
            self._curr_line_manager.append(self._line.indent())
            self._curr_line_manager[-1].__enter__()
            self._line(f'if {trigger} then')

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
        # if stop is None or stop == start:
        #     idx = start
        # else:
        idx = f'{stop} downto {start}'
        return f'{name}({idx})'

    def _emit_switch_start(self, sel: str):
        self._line(f'case? ({sel}) is')

    def _emit_switch_end(self):
        self._line('end case?;')

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
        self._line(f'if {sel} then')

    def _emit_elseif_start(self, sel: str):
        self._line(f'elsif {sel} then')

    def _emit_else(self):
        self._line(f'else')

    def _emit_if_end(self):
        self._line('end if;')

    def _resize_and_sign(self, value: celosia_wire.Wire, width: int, signed: bool = None, ignore_size=False) -> str:
        need_resize = value.width != width

        if isinstance(value, celosia_wire.Const) and need_resize and not ignore_size:
            if value.value < 0:
                value.value += 2**value.width

            if width > value.width:
                extend = value[-1].value if signed else 0
                value.value |= int(str(extend) * (width - value.width), 2) << value.width

            value.width = width
            value.value &= int('1' * width, 2)
            need_resize = False

        value = self._signed(value, signed=signed)
        if need_resize and not ignore_size:
            value = f'resize({value}, {width})'

        return value

    def _signed(self, value: celosia_wire.Wire, signed: bool = None) -> str:
        prefix = ''
        if not signed:
            prefix = 'un'
        if isinstance(value, celosia_wire.Const):
            if value.value < 0:
                value.value += 2**value.width
            if signed is not None:
                if value.value < 2**31:
                    value = f'to_{prefix}signed({value.value}, {value.width})'
                else:
                    value = f"{prefix}signed'({self._represent(value)})"
        else:
            value = f'{prefix}signed({self._represent(value)})'
        return value

    def _to_boolean(self, signal: celosia_wire.Wire) -> str:
        # Const
        # if isinstance(signal, celosia_wire.Const):
        #     return ['true' if signal.value else 'false']

        if isinstance(signal, celosia_wire.Slice):
            wire_rep = self._represent(signal.wire, boolean=False)
            return f'{wire_rep}({signal.start_idx})'

        if isinstance(signal, celosia_wire.Const):
            return 'true' if signal.value else 'false'

        return f'{signal.name}(0)'

    def _operator_repr(self, operator: rtlil.Cell, boolean: bool = False) -> str:
        # TODO: Any issues with constant unary?
        UNARY_OPERATORS = {
            "$neg": "-",
            "$not": "not",
        }
        BINARY_OPERATORS = {
            "$add": '+',
            "$sub": '-',
            "$mul": '*',
            "$divfloor": '/',
            "$modfloor": '%',
            "$and": 'and',
            "$or": 'or',
            "$xor": 'xor',
        }
        SHIFT_OPERATORS = {
            "$shl": 'shift_left',
            "$shr": 'shift_right',
            "$shift": 'shift_right',
            "$sshr": 'shift_right', # TODO: Check sign?
        }

        BOOL_OPERATORS_UNARY = {
            "$reduce_bool": "or",    # TODO: Check
            "$reduce_or": "or",
            "$reduce_and": "and",
            "$reduce_xor": "xor",
        }

        BOOL_OPERATORS_BINARY = {
            "$eq": '=',
            "$ne": '!=',
            "$lt": '<',
            "$gt": '>',
            "$le": '<=',
            "$ge": '>=',
        }

        UNARY_OPERATORS.update(BOOL_OPERATORS_UNARY)
        BINARY_OPERATORS.update(BOOL_OPERATORS_BINARY)

        rhs = None

        target_width = celosia_wire.Cell(operator).width

        if (operator.kind in BOOL_OPERATORS_BINARY | BOOL_OPERATORS_UNARY) and not boolean:
            rhs = f'"1" when {self._operator_repr(operator, boolean=True)} else "0"'

        elif operator.kind in UNARY_OPERATORS:
            operands = []
            for port in ('A',):
                operands.append(self._resize_and_sign(
                    value = self._convert_signals(operator.ports[port]),
                    width = target_width,
                    signed = operator.parameters[f'{port}_SIGNED'],
                    ignore_size = operator.kind != '$neg',
                ))

            rhs = f'{UNARY_OPERATORS[operator.kind]} {operands[0]}'

        elif operator.kind in BINARY_OPERATORS:
            operands = []
            for port in ('A', 'B'):
                operands.append(self._convert_signals(operator.ports[port]))

            if operator.kind in BOOL_OPERATORS_BINARY:
                target_width = max(operand.width for operand in operands)

            for i, port in enumerate(('A', 'B')):
                operands[i] = self._resize_and_sign(
                    value = operands[i],
                    width = target_width,
                    signed = operator.parameters[f'{port}_SIGNED'],
                    ignore_size = operator.kind == '$mul',
                )

            rhs = f' {BINARY_OPERATORS[operator.kind]} '.join(operands)

        elif operator.kind in SHIFT_OPERATORS:
            resize_output = False
            operands = []
            for i, port in enumerate(('A', 'B')):
                operand = self._convert_signals(operator.ports[port])

                ignore_size = True
                if i == 0:
                    if operand.width > target_width:
                        resize_output = True
                    elif operand.width < target_width:
                        ignore_size = False

                operand = self._resize_and_sign(
                    value = operand,
                    width = target_width,
                    signed = operator.parameters[f'{port}_SIGNED'],
                    ignore_size = ignore_size,
                )

                if i == 1:
                    operand = f'to_integer({operand})'

                operands.append(operand)

            rhs = f'{SHIFT_OPERATORS[operator.kind]}({", ".join(operands)})'
            if resize_output:
                rhs = f'{rhs}({target_width-1} downto 0)'

        elif operator.kind == '$mux':
            operands = []
            for i, port in enumerate(('S', 'B', 'A')):
                operand = self._convert_signals(operator.ports[port])

                if i == 0:
                    operand = self._represent(operand, boolean = True)
                else:
                    operand = self._resize_and_sign(
                        value = operand,
                        width = target_width,
                        signed = False,
                        ignore_size=i==0,
                    )
                operands.append(operand)

            rhs = f'std_logic_vector({operands[1]}) when {operands[0]} else std_logic_vector({operands[2]})'

        if rhs is None:
            raise RuntimeError(f"Unknown operator: {operator.kind}")

        return rhs

    def _mem_slice_repr(self, idx: celosia_wire.MemoryIndex):
        return self._slice_repr(idx.name, f'to_integer({self._represent(idx.address)}')

    def _emit_switch(self, switch: rtlil.Switch, comb=True):
        self._strip_unused_cases(switch)

        if not self._is_switch_if(switch):
            for case in switch.cases:
                if not case.patterns:
                    break
            else:
                switch.default()

        return super()._emit_switch(switch, comb=comb, ignore_unused=True)