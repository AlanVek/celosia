from celosia.hdl import HDL
import celosia.backend.signal as celosia_signal
import celosia.backend.module as celosia_module
import celosia.backend.statement as celosia_statement
from textwrap import indent
from amaranth.hdl import ast, ir
from typing import Union

class Verilog(HDL):
    case_sensitive = True
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

    template = """module {name} ({ports}
);{initials}{submodules}{blocks}{assignments}
endmodule
"""

    extension = 'v'
    open_comment = '/* '
    close_comment = ' */'

    def sanitize(self, name: str) -> str:
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

        while name in self.protected:
            name = 'esc_' + name

        if not name:
            name = self.sanitize('unnamed')

        if name[0].isnumeric():
            name = 'esc_' + name

        return name

    def _generate_signal(self, mapping: celosia_signal.Signal, size: bool = True) -> str:
        # TODO: Check what to do with zero-width signals
        # if len(mapping.signal) <= 0:
        #     raise RuntimeError(f"Zero-width mapping {mapping.name} not allowed")

        if not size or len(mapping.signal) <= 1:
            width = ''
        else:
            width = f'[{len(mapping.signal) - 1}:0] '

        if size and isinstance(mapping, celosia_signal.Memory):
            if len(mapping.init) <= 0:
                raise RuntimeError(f"Zero-depth memory {mapping.name} not allowed")
            depth = f' [{len(mapping.init) - 1}:0]'
        else:
            depth = ''

        return f'{width}{mapping.name}{depth}'

    def _generate_port(self, port: celosia_signal.Port) ->str:
        return self._generate_signal(port, size=False)

    def _generate_initial(self, mapping: celosia_signal.Signal) -> str:
        res = ''

        # celosia_signal.Memory ports don't have initials, they're created with the parent signal's celosia_signal.Memory
        if isinstance(mapping, celosia_signal.MemoryPort):
            return res

        reset = None
        dir = ''

        if isinstance(mapping, celosia_signal.Memory):
            type = 'reg'
        else:
            if mapping.static:
                type = 'wire'
            else:
                type = 'reg'
                if mapping.domain is not None:
                    reset = self._parse_rhs(ast.Const(mapping.signal.reset, len(mapping.signal)))

            if isinstance(mapping, celosia_signal.Port):
                dir = 'input' if mapping.direction == 'i' else 'output' if mapping.direction == 'o' else 'inout'
                dir += ' '

        for key, value in mapping.attrs.items():
            if isinstance(value, int):
                value = ast.Const(value, max(32, value.bit_length()))
            res += f'(* {key} = {self._parse_rhs(value, allow_signed=False)} *)\n'

        res += f'{dir}{type} {self._generate_signal(mapping)}'
        if reset is not None:
            res += f' = {reset}'
        res += ';'

        if isinstance(mapping, celosia_signal.Memory):
            res += '\n'.join((
                '',
                'initial begin',
                *[f'{self.tabs()}{mapping.name}[{i}] = {self._parse_rhs(reset)};' for i, reset in enumerate(mapping.init)],
                'end',
            ))

        return res

    def _generate_assignment(self, mapping: celosia_signal.Signal, statement: list[celosia_statement.Statement]) -> str:
        symbol = self._get_symbol(mapping)
        if mapping.static:
            prefix = 'assign '
        else:
            prefix = ''

        start_idx = statement._start_idx
        stop_idx = statement._stop_idx

        if start_idx is None:
            start_idx = 0
        if stop_idx is None:
            stop_idx = len(mapping.signal)

        repr = mapping.name

        size = len(mapping.signal)

        if isinstance(mapping, celosia_signal.MemoryPort):
            repr = f'{repr}[{self._parse_rhs(mapping.index)}]'

        if start_idx != 0 or stop_idx != len(mapping.signal):
            size = max(1, stop_idx - start_idx)
            if size == 1:
                repr = f'{repr}[{start_idx}]'
            else:
                repr = f'{repr}[{stop_idx-1}:{start_idx}]'

        return f'{prefix}{repr} {symbol} {self._parse_rhs(statement.rhs)};'

    def _generate_block(self, mapping: celosia_signal.Signal):
        statements: list[celosia_statement.Statement]= []

        if mapping.reset_statement is not None:
            statements.append(mapping.reset_statement)
        statements.extend(mapping.statements)

        if not statements:
            return ''

        if mapping.domain is None:
            triggers = '*'
        else:
            triggers = '('
            triggers += f'{mapping.domain.clk_edge}edge {self._parse_rhs(mapping.domain.clk)}'

            if mapping.domain.async_reset and mapping.domain.rst is not None:
                triggers += f', posedge {self._parse_rhs(mapping.domain.rst)}'

            triggers += ')'

        return '\n'.join((
            f'always @{triggers} begin',
            indent(self._generate_statements(mapping, statements), self.tabs()),
            'end',
            '', # Add new line at the end to separate blocks
        ))

    def _generate_if(self, mapping: celosia_signal.Signal, statement: celosia_statement.Statement, as_if: list[Union[celosia_statement.Switch.If, celosia_statement.Switch.Else]]):
        if_opening = 'if'
        else_done = False

        ret = ''
        for i, case in enumerate(as_if):
            if else_done:
                raise RuntimeError("New case after 'else'")

            if isinstance(case, celosia_statement.Switch.If):
                opening = if_opening
                if_opening = 'else if'
            elif isinstance(case, celosia_statement.Switch.Else):
                opening = 'else'
                else_done = True

            # if (
            #     (len(case.statements) == 0) or
            #     (len(case.statements) > 1) or
            #     (len(case.statements) == 1 and not isinstance(case.statements[0], Assign))
            # ):
            begin = ' begin'
            end = 'end'
            # else:
            #     begin = end = ''

            if case.test is None:
                ret += f'{opening}{begin}'
            else:
                ret += f'{opening} ({self._parse_rhs(case.test)}){begin}'

            if case.statements:
                case_body = self._generate_statements(mapping, case.statements)
            else:
                case_body = ''

            ret += '\n' + indent(case_body, self.tabs())
            if case_body and end:
                ret += '\n'
            ret += end

            if i < len(as_if) - 1:
                if end:
                    ret += ' '
                else:
                    ret += '\n'

        return ret

    def _generate_switch(self, mapping: celosia_signal.Signal, statement: celosia_statement.Switch):
        if statement.as_if is not None:
            return self._generate_if(mapping, statement, statement.as_if)

        body = []
        for case, statements in statement.cases.items():
            if case is None:
                case = 'default'
            elif '?' in case:
                case = f"{len(statement.test)}'b{case}"
            else:
                case = f"{len(statement.test)}'d{int(case, 2)}"

            if case is None:
                case = 'default'

            # if len(statements) > 1 or (len(statements) == 1 and not isinstance(statements[0], Assign)):
            begin = ' begin'
            end = f'{self.tabs()}end'

            body.append(f'{self.tabs()}{case}:{begin}')

            if statements:
                case_body = self._generate_statements(mapping, statements)
            else:
                case_body = '/* empty */;'

            body.append(indent(case_body, self.tabs(2)))
            # if end:
            body.append(end)

        return '\n'.join((
            f'casez ({self._parse_rhs(statement.test)})',
            *body,
            'endcase',
        ))

    def _generate_submodule(self, submodule: celosia_module.Module):
        res = ''

        res += f'{submodule.type}'

        if submodule.parameters:
            res += ' #(\n'
            for key, value in submodule.parameters.items():
                res += f'{self.tabs()}.{key}({self._parse_rhs(value)}),\n'   # TODO: Check types
            res = res[:-2] + '\n)'

        res += f' {submodule.name} (\n'
        for port in submodule.ports:
            res += f'{self.tabs()}.{port.name}({self._parse_rhs(port.signal, allow_signed=False)}),\n'

        res = res[:-2] + '\n);'

        return res

    def _parse_rhs(self, rhs: Union[ast.Value, int, str, celosia_signal.MemoryPort], allow_signed: bool = True):
        if isinstance(rhs, ast.Const):
            signed = rhs.signed and allow_signed
            value = rhs.value
            if value < 0:
                value += 2**rhs.width

            rhs = f"{max(1, rhs.width)}'h{hex(value)[2:]}"
            if signed:
                rhs = f'$signed({rhs})'

        elif isinstance(rhs, int):
            pass

        elif isinstance(rhs, str):
            rhs = rhs.replace('"', '\\"')
            rhs = f'"{rhs}"'

        elif isinstance(rhs, ast.Signal):
            signed = allow_signed and rhs.shape().signed
            rhs = self.module.signals.get(rhs).name
            if signed:
                rhs = f'$signed({rhs})'
        elif isinstance(rhs, ast.Cat):
            rhs = f"{{ {', '.join(self._parse_rhs(part) for part in rhs.parts[::-1])} }}"
        elif isinstance(rhs, ast.Slice):
            if rhs.start == 0 and rhs.stop >= len(rhs.value):
                rhs = self._parse_rhs(rhs.value)
            else:
                if rhs.stop == rhs.start + 1:
                    idx = rhs.start
                else:
                    idx = f'{rhs.stop-1}:{rhs.start}'

                rhs = f"{self._parse_rhs(rhs.value, allow_signed=False)}[{idx}]"

        elif isinstance(rhs, ast.Operator):
            allow_signed = rhs.operator != 'u'
            parsed = list(map(lambda x: self._parse_rhs(x, allow_signed=allow_signed), rhs.operands))
            if len(rhs.operands) == 1:
                p0 = parsed[0]
                if rhs.operator == '+':
                    rhs = p0
                elif rhs.operator in ('~', '-'):
                    rhs = f'{rhs.operator} {p0}'
                elif rhs.operator == 'b':
                    rhs = f'{p0} != {self._parse_rhs(ast.Const(0, len(rhs.operands[0])))}'
                elif rhs.operator in ('r|', 'r&', 'r^'):
                    rhs = f'{rhs.operator[-1]} {p0}'
                elif rhs.operator == "u":
                    rhs = p0
                elif rhs.operator == "s":
                    if rhs.operands[0].shape().signed:
                        rhs = p0
                    else:
                        rhs = f'$signed({p0})'
                else:
                    raise RuntimeError(f"Unknown operator and operands: {rhs.operator}, {rhs.operands}")
            elif len(rhs.operands) == 2:
                p0, p1 = parsed
                if rhs.operator in ('+', '-', '*', '//', '%', '&', '^', '|'):
                    rhs = f'{p0} {rhs.operator[0]} {p1}'
                elif rhs.operator in ('<', '<=', '==', '!=', '>', '>=', '<<', '>>'):
                    rhs = f'{p0} {rhs.operator} {p1}'
                else:
                    raise RuntimeError(f"Unknown operator and operands: {rhs.operator}, {rhs.operands}")
            elif len(rhs.operands) == 3:
                p0, p1, p2 = parsed
                if rhs.operator == "m":
                    rhs = f'{p0} ? {p1} : {p2}'
                else:
                    raise RuntimeError(f"Unknown operator and operands: {rhs.operator}, {rhs.operands}")
            else:
                raise RuntimeError(f"Unknown operator and operands: {rhs.operator}, {rhs.operands}")

        elif isinstance(rhs, ast.Part):
            if rhs.stride != 1:
                raise RuntimeError("Only Parts with stride 1 supported at end stage!")
            rhs = f'{self._parse_rhs(rhs.value)} >> {self._parse_rhs(rhs.offset)}'
        elif isinstance(rhs, celosia_signal.MemoryPort):
            rhs = f'{self._parse_rhs(rhs.signal)}[{self._parse_rhs(rhs.index)}]'
        else:
            raise ValueError(f"Unknown RHS object detected: {rhs}")

        return rhs

    @staticmethod
    def _get_symbol(mapping: ast.Signal) -> str:
        if mapping.static or mapping.domain is None:
            res ='='
        else:
            res = '<='
        return res


def convert(
    module: Union[ir.Fragment, ir.Elaboratable],
    name: str = 'top',
    ports: list[ast.Signal] = None,
    platform = None,
    spaces: int = 4,
    fragment_prepare: bool = True,
):
    return Verilog(
        spaces = spaces,
    ).convert(
        module,
        name = name,
        ports = ports,
        platform = platform,
        fragment_prepare = fragment_prepare,
    )
