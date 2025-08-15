from pyhdl.hdl import HDL
import pyhdl.backend.signal as pyhdl_signal
import pyhdl.backend.statement as pyhdl_statement
from textwrap import indent
from amaranth.hdl import ast, ir, dsl

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

    def _generate_signal(self, mapping: pyhdl_signal.Signal, size: bool = True) -> str:
        # TODO: Check what to do with zero-width signals
        # if len(mapping.signal) <= 0:
        #     raise RuntimeError(f"Zero-width mapping {mapping.signal.name} not allowed")

        if not size or len(mapping.signal) <= 1:
            width = ''
        else:
            width = f'[{len(mapping.signal) - 1}:0] '

        if size and isinstance(mapping, pyhdl_signal.Memory):
            if len(mapping.init) <= 0:
                raise RuntimeError(f"Zero-depth memory {mapping.signal.name} not allowed")
            depth = f' [{len(mapping.init) - 1}:0]'
        else:
            depth = ''

        return f'{width}{mapping.signal.name}{depth}'

    def _generate_port(self, port: pyhdl_signal.Port) ->str:
        return self._generate_signal(port, size=False)

    def _generate_initial(self, mapping: pyhdl_signal.Signal) -> str:
        res = ''

        # pyhdl_signal.Memory ports don't have initials, they're created with the parent signal's pyhdl_signal.Memory
        if isinstance(mapping, pyhdl_signal.MemoryPort):
            return res

        if isinstance(mapping, pyhdl_signal.Signal):
            reset = None
            dir = ''

            if isinstance(mapping, pyhdl_signal.Memory):
                type = 'reg'
            else:
                if mapping.static:
                    type = 'wire'
                else:
                    type = 'reg'
                    if mapping.domain is not None:
                        reset = self._parse_rhs(ast.Const(mapping.signal.reset, len(mapping.signal)))

                if isinstance(mapping, pyhdl_signal.Port):
                    dir = 'input' if mapping.direction == 'i' else 'output' if mapping.direction == 'o' else 'inout'
                    dir += ' '

            res += f'{dir}{type} {self._generate_signal(mapping)}'
            if reset is not None:
                res += f' = {reset}'
            res += ';'

        if isinstance(mapping, pyhdl_signal.Memory):
            res += '\n'.join((
                '',
                'initial begin',
                *[f'{self.tabs()}{mapping.signal.name}[{i}] = {self._parse_rhs(reset)};' for i, reset in enumerate(mapping.init)],
                'end',
            ))

        return res

    def _generate_assignment(self, mapping: pyhdl_signal.Signal, statement: list[pyhdl_statement.Statement]) -> str:
        symbol = self._get_symbol(mapping)
        if mapping.static:
            prefix = 'assign '
        else:
            prefix = ''

        start_idx = statement._start_idx
        stop_idx = statement._stop_idx

        repr = mapping.signal.name

        size = len(mapping.signal)

        if isinstance(mapping, pyhdl_signal.MemoryPort):
            repr = f'{repr}[{self._parse_rhs(mapping.index)}]'

        elif start_idx is not None and stop_idx is not None:
            if start_idx != 0 or stop_idx != len(mapping.signal):
                size = max(1, stop_idx - start_idx)
                if size == 1:
                    repr = f'{repr}[{start_idx}]'
                else:
                    repr = f'{repr}[{stop_idx-1}:{start_idx}]'
        elif start_idx is not None or stop_idx is not None:
            raise RuntimeError(f"Invalid assignment, start_idx and stop_idx must be both None or have value ({start_idx} - {stop_idx})")

        return f'{prefix}{repr} {symbol} {self._parse_rhs(statement.rhs)};'

    def _generate_block(self, mapping: pyhdl_signal.Signal):
        statements: list[pyhdl_statement.Statement]= []

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

    def _generate_if(self, mapping, statement, as_if):
        if_opening = 'if'
        else_done = False

        ret = ''
        for i, case in enumerate(as_if):
            if else_done:
                raise RuntimeError("New case after 'else'")

            if isinstance(case, pyhdl_statement.Switch.If):
                opening = if_opening
                if_opening = 'else if'
            elif isinstance(case, pyhdl_statement.Switch.Else):
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

    def _generate_switch(self, mapping, statement):
        if statement.as_if is not None:
            return self._generate_if(mapping, statement, statement.as_if)

        body = []
        for case, statements in statement.cases.items():
            if isinstance(case, str):
                try:
                    case = int(case, 2)
                    case = f"{len(statement.test)}'d{case}"
                except ValueError:
                    case = f"{len(statement.test)}'b{case}"

            elif isinstance(case, int):
                case = f"{len(statement.test)}'d{case}"

            elif case is not None:
                raise RuntimeError(f"Unknown case for switch: {case}")

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

    def _generate_submodule(self, submodule, ports, parameters):
        res = ''

        res += f'{submodule.type}'

        if parameters:
            res += ' #(\n'
            for key, value in parameters.items():
                res += f'{self.tabs()}.{key}({self._parse_rhs(value)}),\n'   # TODO: Check types
            res = res[:-2] + '\n)'

        res += f' {submodule.name} (\n'
        for key, value in ports.items():
            res += f'{self.tabs()}.{key}({self._parse_rhs(value, allow_signed=False)}),\n'

        res = res[:-2] + '\n);'

        return res

    def _parse_rhs(self, rhs, allow_signed=True):
        if isinstance(rhs, ast.Const):
            signed = rhs.signed
            value = rhs.value
            if value < 0:
                value += 2**rhs.width

            rhs = f"{max(1, rhs.width)}'h{hex(value)[2:]}"
            if signed:
                rhs = f'$signed({rhs})'

        elif isinstance(rhs, int):
            pass

        elif isinstance(rhs, str):
            rhs = f'"{rhs}"'

        elif isinstance(rhs, ast.Signal):
            signed = allow_signed and rhs.shape().signed
            rhs = rhs.name
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
        elif isinstance(rhs, pyhdl_signal.MemoryPort):
            rhs = f'{self._parse_rhs(rhs.signal)}[{self._parse_rhs(rhs.index)}]'
        else:
            raise ValueError("Unknown RHS object detected: {}".format(rhs))

        return rhs

    @staticmethod
    def _get_symbol(mapping: ast.Signal) -> str:
        if mapping.static or mapping.domain is None:
            res ='='
        else:
            res = '<='
        return res


def convert(
    module: dsl.Module | ir.Fragment,
    name: str = 'top',
    ports: list[ast.Signal] = None,
    platform = None,
    spaces: int = 4,
):
    print('HERE')
    return Verilog(
        spaces = spaces,
    ).convert(
        module,
        name = name,
        ports = ports,
        platform = platform,
    )
