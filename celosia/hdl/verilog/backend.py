from celosia.hdl.backend import Module as BaseModule
from typing import Any, Union
from amaranth.back import rtlil
import re

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

    def _sanitize(self, name: str) -> str:
        name = super()._sanitize(name).strip()

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

        while name in self.protected:
            name = 'esc_' + name

        if not name:
            name = self._sanitize('unnamed')

        if name[0].isnumeric():
            name = 'esc_' + name

        return name

    @staticmethod
    def _const_repr(width, value):

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
    def _concat(cls, parts):
        return f'{{ {", ".join(parts)} }}'

    def _emit_assignment_lhs_rhs(self, lhs: str, rhs: str, symbol = '=', prefix=None):
        if prefix is None:
            prefix = ''
        elif prefix:
            prefix += ' '

        with self._line.indent():
            self._line(f'{prefix}{self._get_signal_name(lhs)} {symbol} {self._get_signal_name(rhs)};')

    def _emit_assignment(self, assignment: rtlil.Assignment):
        self._emit_assignment_lhs_rhs(assignment.lhs, assignment.rhs, prefix='assign')

    def _emit_process_assignment(self, assignment: rtlil.Assignment):
        self._emit_assignment_lhs_rhs(assignment.lhs, assignment.rhs)

    def _emit_ff_assignment(self, assignment: rtlil.Assignment):
        self._emit_assignment_lhs_rhs(assignment.lhs, assignment.rhs, symbol = '<=')

    def _emit_submodule(self, submodule: rtlil.Cell):
        print('Emit submodule:', submodule.name, submodule.kind)
        pass

    def _emit_operator(self, operator: rtlil.Cell):
        print('Emit operator:', operator.name, operator.kind, operator.ports)
        pass

    def _emit_signal(self, signal: rtlil.Wire):
        super()._emit_signal(signal)
        with self._line.indent():
            type = 'reg' if self._signal_is_reg(signal) else 'wire'
            init = self._get_initial(signal)

            if signal.width <= 1:
                width = ''
            else:
                width = f'[{signal.width - 1}:0] '

            if signal.port_kind is None:
                dir = ''
            else:
                dir = f'{signal.port_kind} '

            if init is None:
                reset = ''
            else:
                reset = f' = {init}'

            for key, attr in signal.attributes.items():
                self._line(f'(* {key} = {self._const(attr)} *)')

            self._line(f'{dir}{type} {width}{signal.name}{reset};')

    def _emit_memory(self, memory: rtlil.Memory):
        print('Emit memory:', memory.name, memory.depth, memory.width)
        pass

    def _emit_module_and_ports(self, ports: list["rtlil.Wire"]):
        self._collect_process_signals(self._emitted_processes)

        self._line(f"module {self.name} (")
        with self._line.indent():
            for i, port in enumerate(ports):
                self._line(port.name + ('' if i >= len(ports) - 1 else ','))
        self._line(');')

        for i, port in enumerate(ports):
            self._emit_signal(port)

    def _emit_flip_flop_start(self, clock: str, polarity: bool, arst: str = None, arst_polarity = False) -> str:
        ret = super()._emit_flip_flop_start(clock, polarity, arst, arst_polarity)

        polarity = 'pos' if polarity else 'neg'
        arst_polarity = 'pos' if arst_polarity else 'neg'
        trigger = f'always @ ({polarity}edge {clock}'

        if arst is not None:
            trigger += f', {arst_polarity}edge {arst}'

        self._line(trigger + ') begin')

        return ret

    def _emit_flip_flop_end(self, ff_id: str):
        self._line('end')

    def _signal_is_reg(self, signal: rtlil.Wire):
        return 'init' in signal.attributes or signal.name in self._regs

    def _emit_process_start(self) -> str:
        ret = super()._emit_process_start()
        self._line('always @* begin')
        return ret

    def _emit_process_end(self, p_id: str):
        self._line('end')

    @classmethod
    def _get_slice(cls, name: str, start: int, stop: int):
        if stop == start:
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

    def _get_raw_signals(self, signal: str) -> set[str]:
        ret = set()

        if signal is None:
            return ret

        slice_pattern = re.compile(r'(.*?) \[(.*?)\]')

        if signal.startswith('{') and signal.endswith('}'):
            signal = signal[1:-1].strip()

        while signal:
            slice_match = slice_pattern.match(signal)
            if slice_match is not None:
                name = slice_match.group(1)
                ret.add(name)
                signal = signal[slice_match.end() + 1:]
                continue

            space_idx = signal.find(' ')
            if space_idx < 0:
                space_idx = len(signal)
                ret.add(signal)
            else:
                ret.add(signal[:space_idx])

            signal = signal[space_idx + 1:]

        return ret

    def _collect_lhs(self, assignment: Union[rtlil.Assignment, rtlil.Switch, rtlil.Case]) -> set[str]:
        ret = set()


        # TODO: We can probably break early if LHS is never a concatenation
        if isinstance(assignment, rtlil.Assignment):
            print('Collect:', assignment.lhs, self._get_raw_signals(assignment.lhs))
            ret.update(self._get_raw_signals(assignment.lhs))

        elif isinstance(assignment, rtlil.Switch):
            for case in assignment.cases:
                ret.update(self._collect_lhs(case))

        elif isinstance(assignment, rtlil.Case):
            for content in assignment.contents:
                ret.update(self._collect_lhs(content))

        return ret

    def _collect_process_signals(self, processes: list[rtlil.Process]):
        for process in processes:
            for content in process.contents:
                self._regs.update(self._collect_lhs(content))
