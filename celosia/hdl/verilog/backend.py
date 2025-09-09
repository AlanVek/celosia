from celosia.hdl.module import Module as BaseModule
from typing import Any, Union
from amaranth.back import rtlil

class VerilogModule(BaseModule):

    protected = [
        'always',         'and',            'assign',         'begin',
        'buf',            'bufif0',         'bufif1',         'case',
        'casex',          'casez',          'cmos',           'config',
        'deassign',       'default',        'defparam',       'design',
        'disable',        'edge',           'else',           'end',
        'endcase',        'endfunction',    'endmodule',      'endprimitive',
        'endspecify',     'endtable',       'endtask',        'event',
        'for',            'force',          'forever',        'fork',
        'function',       'highz0',         'highz1',         'if',
        'ifnone',         'initial',        'inout',          'input',
        'integer',        'join',           'large',          'localparam',
        'macromodule',    'medium',         'module',         'nand',
        'negedge',        'nmos',           'nor',            'not',
        'notif0',         'notif1',         'or',             'output',
        'parameter',      'pmos',           'posedge',        'primitive',
        'pull0',          'pull1',          'pulldown',       'pullup',
        'rcmos',          'real',           'realtime',       'reg',
        'release',        'repeat',         'rnmos',          'rpmos',
        'rtran',          'rtranif0',       'rtranif1',       'scalared',
        'signed',         'small',          'specify',        'specparam',
        'strong0',        'strong1',        'supply0',        'supply1',
        'table',          'task',           'time',           'tran',
        'tranif0',        'tranif1',        'tri',            'tri0',
        'tri1',           'triand',         'trior',          'trireg',
        'unsigned',       'vectored',       'wait',           'wand',
        'weak0',          'weak1',          'while',          'wire',
        'wor',            'xnor',           'xor',
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._regs: set[str] = set()

    @classmethod
    def _const(cls, value: Any):
        if isinstance(value, str):
            value = value.replace('"', '\\"')
            return f'"{value}"'
        return super()._const(value)

    @classmethod
    def sanitize(cls, name: str) -> str:
        name = super().sanitize(name).strip()

        # TODO: Update sanitization for Verilog

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

        while name in cls.protected:
            name = 'esc_' + name

        if not name:
            name = cls.sanitize('unnamed')

        if name[0].isnumeric():
            name = 'esc_' + name

        return name

    @staticmethod
    def _const_repr(width, value, init=False):

        if isinstance(value, str):
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
        return f'{{ {", ".join(parts[::-1])} }}'

    def _emit_assignment_lhs_rhs(self, lhs: str, rhs: str, symbol = '=', prefix=None, parse=True):
        if prefix is None:
            prefix = ''
        elif prefix:
            prefix += ' '

        if parse:
            lhs = self._represent(lhs)
            rhs = self._represent(rhs)

        self._line(f'{prefix}{lhs} {symbol} {rhs};')

    def _emit_assignment(self, assignment: rtlil.Assignment):
        self._emit_assignment_lhs_rhs(assignment.lhs, assignment.rhs, prefix='assign')

    def _emit_process_assignment(self, assignment: rtlil.Assignment, comb = True):
        self._emit_assignment_lhs_rhs(assignment.lhs, assignment.rhs, symbol = '=' if comb else '<=')

    def _emit_operator_assignment(self, assignment: rtlil.Assignment, comb = True):
        self._emit_assignment_lhs_rhs(assignment.lhs, assignment.rhs, symbol = '=' if comb else '<=', prefix='assign')

    def _emit_submodule_post(self, submodule: rtlil.Cell, instance: bool):
        with self._line.indent():
            line = f'{submodule.kind} '
            if submodule.parameters:
                line += '#('
                self._line(line)
                with self._line.indent():
                    for i, (name, value) in enumerate(submodule.parameters.items()):
                        sep = "," if i < len(submodule.parameters) - 1 else ""
                        self._line(f'.{name}({self._const(value)}){sep}')
                line = ') '

            line += str(submodule.name)

            if submodule.ports:
                line += ' ('
                self._line(line)
                with self._line.indent():
                    for i, (name, value) in enumerate(submodule.ports.items()):
                        sep = "," if i < len(submodule.ports) - 1 else ""
                        self._line(f'.{name}({self._represent(value)}){sep}')
                line = ')'

            self._line(f'{line};')

    def _emit_signal(self, signal: rtlil.Wire):
        with self._line.indent():
            init = self._get_initial(signal)
            if self._signal_is_reg(signal):
                type = 'reg'
            else:
                type = 'wire'
                init = None

            width = '' if signal.width <= 1 else f'[{signal.width - 1}:0] '
            dir = '' if signal.port_kind is None else f'{signal.port_kind} '
            depth = len(init) if isinstance(init, list) else 0

            for key, attr in signal.attributes.items():
                self._line(f'(* {key} = {self._const(attr)} *)')

            if depth:
                self._line(f'{dir}{type} {width}{signal.name} [{depth-1}:0];')
                self._line('initial begin')
                with self._line.indent():
                    for i, value in enumerate(init):
                        self._emit_assignment_lhs_rhs(self._slice_repr(signal.name, i, i), value, parse=False)
                self._line('end')
            else:
                reset = '' if init is None else f' = {init}'
                self._line(f'{dir}{type} {width}{signal.name}{reset};')

    def _emit_pre_callback(self):
        self._collect_process_signals()

    def _emit_module_definition(self):
        self._line(f"module {self.name} (")
        with self._line.indent():
            for i, port in enumerate(self._emitted_ports):
                self._line(port.name + ('' if i >= len(self._emitted_ports) - 1 else ','))
        self._line(');')

    def _signal_is_reg(self, signal: rtlil.Wire):
        return signal.name in self._regs

    def _emit_process_start(self, clock = None, polarity: bool = True, arst: str = None, arst_polarity = False) -> str:
        ret = super()._emit_process_start(clock, polarity, arst, arst_polarity)

        if clock is None:
            trigger = '*'
        else:
            trigger = f'({"pos" if polarity else "neg"}edge {self._represent(clock)}'
            if arst is not None:
                trigger += f', {"pos" if arst_polarity else "neg"}edge {self._represent(arst)}'
            trigger += ')'

        self._line(f'always @ {trigger} begin')

        return ret

    def _emit_process_end(self, p_id: str, comb=True):
        self._line('end')

    @classmethod
    def _slice_repr(cls, name: str, start: int, stop: int=None) -> str:
        if stop is None or stop == start:
            idx = start
        else:
            idx = f'{stop}:{start}'
        return f'{name}[{idx}]'

    def _emit_switch_start(self, sel: str):
        self._line(f'casez ({sel})')

    def _emit_switch_end(self):
        self._line('endcase')

    def _case_patterns(self, pattern: tuple[str, ...]) -> str:
        return ', '.join(p.replace('-', '?') for p in pattern)

    def _case_default(self) -> str:
        return 'default'

    def _emit_case_start(self, pattern: str):
        self._line(f'{pattern}: begin')

    def _emit_case_end(self):
        self._line('end')

    def _emit_module_end(self):
        self._line('endmodule')

    def _emit_if_start(self, sel: str):
        self._line(f'if ({sel}) begin')

    def _emit_elseif_start(self, sel: str):
        self._line(f'end else if ({sel}) begin')

    def _emit_else(self):
        self._line(f'end else begin')

    def _emit_if_end(self):
        self._line('end')

    def _collect_lhs(self, assignment: Union[rtlil.Assignment, rtlil.Switch, rtlil.Case]) -> set[str]:
        ret = set()

        # TODO: We can probably break early if LHS is never a concatenation
        if isinstance(assignment, rtlil.Assignment):
            ret.update(wire.name for wire in self._get_raw_signals(assignment.lhs))

        elif isinstance(assignment, rtlil.Switch):
            for case in assignment.cases:
                ret.update(self._collect_lhs(case))

        elif isinstance(assignment, rtlil.Case):
            for content in assignment.contents:
                ret.update(self._collect_lhs(content))

        return ret

    def _collect_process_signals(self):
        for process, _ in self._emitted_processes:
            for content in process.contents:
                self._regs.update(self._collect_lhs(content))

        for flip_flop in self._emitted_flip_flops:
            self._regs.update(wire.name for wire in self._get_raw_signals(flip_flop.ports['Q']))

        for memory in self._emitted_memories.values():
            self._regs.add(memory.name)

    def _signed(self, value) -> str:
        return f'$signed({value})'

    def _operator_repr(self, operator: rtlil.Cell, boolean = False) -> str:
        # TODO: Any issues with constant unary?
        UNARY_OPERATORS = {
            "$neg": "-",
            "$not": "~",
            "$reduce_bool": "|",    # TODO: Check
            "$reduce_or": "|",
            "$reduce_and": "&",
            "$reduce_xor": "^",
        }
        BINARY_OPERATORS = {
            "$add": '+',
            "$sub": '-',
            "$mul": '*',
            "$divfloor": '/',
            "$modfloor": '%',
            "$shl": '<<',
            "$shr": '>>',
            "$sshr": '>>', # TODO: Check sign?
            "$and": '&',
            "$or": '|',
            "$xor": '^',
            "$eq": '==',
            "$ne": '!=',
            "$lt": '<',
            "$gt": '>',
            "$le": '<=',
            "$ge": '>=',
            "$shift": '>>',
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
