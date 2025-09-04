from celosia.hdl.backend import Module as BaseModule
from typing import Any
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
        # TODO: Check signed
        return f"{int(width)}'h{hex(int(value))[2:]}"

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
        print('Assign:', assignment.lhs, '--', assignment.rhs)
        self._emit_assignment_lhs_rhs(assignment.lhs, assignment.rhs, prefix='assign')

    def _emit_switch(self, switch: rtlil.Switch):
        print('Emit switch:', switch.sel, switch.cases)

    def _emit_submodule(self, submodule: rtlil.Cell):
        print('Emit submodule:', submodule.name, submodule.kind)
        pass

    def _emit_operator(self, operator: rtlil.Cell):
        print('Emit operator:', operator.name, operator.kind, operator.ports)
        pass

    def _emit_signal(self, signal: rtlil.Wire, direction=None):
        super()._emit_signal(signal)
        with self._line.indent():
            type = 'reg' if self._signal_is_reg(signal) else 'wire'
            init = self._get_initial(signal)

            if signal.width <= 1:
                width = ''
            else:
                width = f'[{signal.width - 1}:0] '

            if direction is None:
                dir = ''
            else:
                dir = f'{direction} '

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
        self._line(f"module {self.name} (")
        with self._line.indent():
            for i, port in enumerate(ports):
                self._line(port.name + ('' if i >= len(ports) - 1 else ','))
        self._line(');')

        for i, port in enumerate(ports):
            self._emit_signal(port, direction = port.port_kind)

    def _emit_flip_flop(self, flip_flop: rtlil.Cell):
        print('Emit flip_flop:', flip_flop.name, flip_flop.kind, flip_flop.ports)

        with self._line.indent():
            data = flip_flop.ports['D']
            clock = flip_flop.ports['CLK']
            out = flip_flop.ports['Q']
            polarity = 'pos' if flip_flop.parameters['CLK_POLARITY'] else 'neg'

            arst = flip_flop.ports.get('ARST', None)
            arst_polarity = 'pos' if flip_flop.parameters.get('ARST_POLARITY', True) else 'neg'
            arst_value= flip_flop.parameters.get('ARST_VALUE', None)

            trigger = f'always @ ({polarity}edge {clock}'

            if arst is not None:
                trigger += f', {arst_polarity}edge {arst}'

            self._line(trigger + ') begin')

            if arst is None:
                self._emit_assignment_lhs_rhs(out, data, symbol='<=')
            else:
                if arst_value is None:
                    raise RuntimeError("Missing arst value for async reset")

                switch = rtlil.Switch(arst)   # TODO: Check negated? Or not worth it?
                reset_case = switch.case('1' if arst_polarity == 'pos' else '0')
                reset_case.assign(out, arst_value)

                default = switch.default()
                default.assign(out, data)

                self._emit_switch(switch)

    def _emit_module_end(self):
        pass

    def _emit_connections(self):
        pass

    def _signal_is_reg(self, signal: rtlil.Wire):
        # TODO: Check for processes and operators
        if 'init' in signal.attributes:
            return True

        return False
