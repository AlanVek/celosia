from amaranth.hdl import ast, ir
from textwrap import dedent, indent

# TODO: If we find multiple signals with same statements, maybe we can merge them into one!

class HDL:
    case_sensitive = False
    spaces = 4

    @classmethod
    def sanitize(cls, name):
        return name

    @classmethod
    def convert(cls, fragment, name='top', ports=None, platform=None):
        m = Module(name, fragment, hdl=cls)
        m.prepare(ports, platform)

        return cls._convert_module(m)

    @classmethod
    def _convert_module(cls, module):
        return ''

    @classmethod
    def tabs(cls, n=1):
        if n <= 0:
            return ''

        return ' ' * (cls.spaces * n)

class VHDL(HDL):
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

    template = """library ieee;
        use ieee.std_logic_1164.all;
        use ieee.numeric_std.all;
        use ieee.numeric_std_unsigned.all;

        entity {name} is
            port (
{port_block}
            );
        end {name};

        architecture rtl of {name} is
{components_block}
{constants_block}
{types_block}
{signal_block}
        begin
{assignment_block}
{submodules_block}
{blocks_block}
        end rtl;
        """

    @classmethod
    def sanitize(cls, name):
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

        return name

    @classmethod
    def _convert_module(cls, module):
        pass

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

    template = """module {name} (
{port_block}
);
{initials_block}
{submodules_block}
{blocks_block}
{assignment_block}
endmodule
"""

    @classmethod
    def sanitize(cls, name):
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

    @classmethod
    def _convert_module(cls, module):
        if module.empty:
            return ''

        port_block, initial_block, assignment_block, blocks_block = cls._parse_signals(module)
        submodules_block = cls._generate_submodule_blocks(module)

        res = [cls.template.format(
            name = module.type,
            port_block = port_block,
            initials_block = initial_block,
            assignment_block = assignment_block,
            submodules_block = submodules_block,
            blocks_block = blocks_block,
        )]
        for submodule, _ in module.submodules:
            if isinstance(submodule, InstanceModule):
                continue
            res.append(cls._convert_module(submodule))

        return '\n'.join(res)

    @classmethod
    def _parse_signals(cls, module):
        port_block = []
        initial_block = []
        assignment_block = []
        blocks_block = []

        for mapping in module._signals.values():
            if not len(mapping.signal):
                continue

            initial_block.append(cls._generate_initial(mapping))
            if isinstance(mapping, Port):
                port_block.append(f'{cls._generate_one_port(mapping)}')

                if mapping.direction == 'i':
                    continue

            if mapping.static:
                if mapping.statements:
                    statement = mapping.statements[0]
                elif mapping.reset_statement is not None:
                    statement = mapping.reset_statement
                else:
                    continue
                assignment_block.append(f'assign {cls._generate_one_assignment(mapping, statement, symbol="=")};')
            else:
                blocks_block.append(f'{cls._generate_one_block(mapping, module)}')

        port_block = ',\n'.join(port_block)
        initial_block = '\n'.join(initial_block)
        assignment_block = '\n'.join(assignment_block)
        blocks_block = '\n'.join(blocks_block)
        blocks = (port_block, initial_block, assignment_block, blocks_block)
        return (indent(block, cls.tabs()) for block in blocks)

    @classmethod
    def _generate_initial(cls, mapping):
        res = ''

        # Memory ports don't have initials, they're created with the parent signal's Memory
        if isinstance(mapping, MemoryPort):
            return res

        if isinstance(mapping, Signal):
            reset = None
            dir = ''

            if isinstance(mapping, Memory):
                type = 'reg'
            else:
                if mapping.static:
                    type = 'wire'
                else:
                    type = 'reg'
                    if mapping.domain is not None:
                        reset = cls._parse_rhs(ast.Const(mapping.signal.reset, len(mapping.signal)))

                if isinstance(mapping, Port):
                    dir = 'input' if mapping.direction == 'i' else 'output' if mapping.direction == 'o' else 'inout'
                    dir += ' '

            res += f'{dir}{type} {cls._generate_one_signal(mapping)}'
            if reset is not None:
                res += f' = {reset}'
            res += ';'

        if isinstance(mapping, Memory):
            res += '\ninitial begin\n'
            for i, reset in enumerate(mapping.init):
                res += f'{cls.tabs()}{mapping.signal.name}[{i}] = {cls._parse_rhs(reset)};\n'
            res += 'end'

        return res

    @staticmethod
    def _generate_one_signal(mapping, size=True):
        # TODO: Check what to do with zero-width signals
        # if len(mapping.signal) <= 0:
        #     raise RuntimeError(f"Zero-width mapping {mapping.signal.name} not allowed")

        if not size or len(mapping.signal) <= 1:
            width = ''
        else:
            width = f'[{len(mapping.signal) - 1}:0] '

        if size and isinstance(mapping, Memory):
            if len(mapping.init) <= 0:
                raise RuntimeError(f"Zero-depth memory {mapping.signal.name} not allowed")
            depth = f' [{len(mapping.init) - 1}:0]'
        else:
            depth = ''

        return f'{width}{mapping.signal.name}{depth}'

    @classmethod
    def _generate_one_port(cls, port):
        return cls._generate_one_signal(port, size=False)
        dir = 'input' if port.direction == 'i' else 'output' if port.direction == 'o' else 'inout'
        return f'{dir} {cls._generate_one_signal(port, size=False)}'

    @classmethod
    def _generate_one_assignment(cls, mapping, statement, symbol):
        start_idx = statement._start_idx
        stop_idx = statement._stop_idx

        repr = mapping.signal.name

        size = len(mapping.signal)

        if isinstance(mapping, MemoryPort):
            repr = f'{repr}[{cls._parse_rhs(mapping.index)}]'

        elif start_idx is not None and stop_idx is not None:
            if start_idx != 0 or stop_idx != len(mapping.signal):
                size = max(1, stop_idx - start_idx)
                if size == 1:
                    repr = f'{repr}[{start_idx}]'
                else:
                    repr = f'{repr}[{stop_idx-1}:{start_idx}]'
        elif start_idx is not None or stop_idx is not None:
            raise RuntimeError(f"Invalid assignment, start_idx and stop_idx must be both None or have value ({start_idx} - {stop_idx})")

        return f'{repr} {symbol} {cls._parse_rhs(statement.rhs)}'

    @classmethod
    def _generate_one_block(cls, mapping, module):

        statements = []

        if mapping.reset_statement is not None:
            statements.append(mapping.reset_statement)
        statements.extend(mapping.statements)

        if not statements:
            return ''

        if mapping.domain is None:
            triggers = '*'
            symbol = '='
        else:
            triggers = '('
            domain = module.domains.get(mapping.domain, None)
            if domain is None:
                raise RuntimeError(f"Unknown domain for module {module.name}: {mapping.domain}")

            triggers += f'{domain.clk_edge}edge {cls._parse_rhs(domain.clk)}'

            if domain.async_reset and domain.rst is not None:
                triggers += f', posedge {cls._parse_rhs(domain.rst)}'

            triggers += ')'
            symbol = '<='

        header = f'always @{triggers} begin'
        blocks = indent(cls._generate_statements(mapping, statements, symbol=symbol), cls.tabs())
        footer = 'end'

        return dedent(f"{header}\n{blocks}\n{footer}\n")

    @classmethod
    def _generate_statements(cls, mapping, statements, symbol):
        res = []
        for statement in statements:
            res.append(cls._generate_one_statement(mapping, statement, symbol))
        return '\n'.join(res)

    @classmethod
    def _generate_one_statement(cls, mapping, statement, symbol):
        res = []

        if isinstance(statement, Assign):
            res.append(f"{cls._generate_one_assignment(mapping, statement, symbol=symbol)};")

        elif isinstance(statement, Switch):
            res.append(f"{cls._generate_switch(mapping, statement, symbol=symbol)}")

        else:
            raise RuntimeError(f"Unknown statement: {statement}")

        return '\n'.join(res)

    @classmethod
    def _generate_if(cls, mapping, statement, as_if, symbol):
        if_opening = 'if'
        else_done = False

        ret = ''
        for i, case in enumerate(as_if):
            if else_done:
                raise RuntimeError("New case after 'else'")

            if isinstance(case, Switch.If):
                opening = if_opening
                if_opening = 'else if'
            elif isinstance(case, Switch.Else):
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
                ret += f'{opening} ({cls._parse_rhs(case.test)}){begin}'

            if case.statements:
                case_body = cls._generate_statements(mapping, case.statements, symbol=symbol)
            else:
                case_body = ''

            ret += '\n' + indent(case_body, cls.tabs())
            if case_body and end:
                ret += '\n'
            ret += end

            if i < len(as_if) - 1:
                if end:
                    ret += ' '
                else:
                    ret += '\n'

        return ret

    @classmethod
    def _generate_switch(cls, mapping, statement, symbol):
        if statement.as_if is not None:
            return cls._generate_if(mapping, statement, statement.as_if, symbol)

        header = f'casez ({cls._parse_rhs(statement.test)})'

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

            if len(statements) > 1 or (len(statements) == 1 and not isinstance(statements[0], Assign)):
                begin = ' begin'
                end = f'{cls.tabs()}end'
            else:
                begin = end = ''

            body.append(f'{cls.tabs()}{case}:{begin}')

            if statements:
                case_body = cls._generate_statements(mapping, statements, symbol=symbol)
            else:
                case_body = '/* empty */;'

            body.append(indent(case_body, cls.tabs(2)))
            if end:
                body.append(end)

        body = '\n'.join(body)
        footer = 'endcase'

        return dedent(f'{header}\n{body}\n{footer}')

    @classmethod
    def _generate_submodule_blocks(cls, module):
        res = ''

        for submodule, ports in module.submodules:
            params = {}
            if not isinstance(submodule, InstanceModule):
                if ports is not None:
                    raise RuntimeError(f"Found invalid submodule configuration for submodule {submodule.name} of module {module.name}")

                if submodule.empty:
                    continue

                ports = {}
                for port in submodule.ports:
                    if not len(port.signal):
                        continue
                    # if port.signal not in module._signals:
                    #     raise RuntimeError(f"Found port {port.signal.name} of submodule {name} which is not a signal of {module.name}")
                    ports[port.signal.name] = port.signal

            else:
                if ports is None:
                    raise RuntimeError(f"Found invalid submodule configuration for submodule {submodule.name} of module {module.name}")

                params.update(submodule.parameters)

            res += f'{submodule.type}'

            if params:
                res += ' #(\n'
                for key, value in params.items():
                    res += f'{cls.tabs()}.{key}({cls._parse_rhs(value)}),\n'   # TODO: Check types
                res = res[:-2] + '\n)'

            res += f' {submodule.name} (\n'
            for key, value in ports.items():
                res += f'{cls.tabs()}.{key}({cls._parse_rhs(value, allow_signed=False)}),\n'

            res = res[:-2] + '\n);\n'

        return indent(res, cls.tabs())

    @classmethod
    def _parse_rhs(cls, rhs, allow_signed=True):
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
            rhs = f"{{ {', '.join(cls._parse_rhs(part) for part in rhs.parts[::-1])} }}"
        elif isinstance(rhs, ast.Slice):
            if rhs.start == 0 and rhs.stop >= len(rhs.value):
                rhs = cls._parse_rhs(rhs.value)
            else:
                if rhs.stop == rhs.start + 1:
                    idx = rhs.start
                else:
                    idx = f'{rhs.stop-1}:{rhs.start}'

                rhs = f"{cls._parse_rhs(rhs.value, allow_signed=False)}[{idx}]"

        elif isinstance(rhs, ast.Operator):
            allow_signed = rhs.operator != 'u'
            parsed = list(map(lambda x: cls._parse_rhs(x, allow_signed=allow_signed), rhs.operands))
            if len(rhs.operands) == 1:
                p0 = parsed[0]
                if rhs.operator == '+':
                    rhs = p0
                elif rhs.operator in ('~', '-'):
                    rhs = f'{rhs.operator} {p0}'
                elif rhs.operator == 'b':
                    rhs = f'{p0} != {cls._parse_rhs(ast.Const(0, len(rhs.operands[0])))}'
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
            rhs = f'{cls._parse_rhs(rhs.value)} >> {cls._parse_rhs(rhs.offset)}'
        elif isinstance(rhs, MemoryPort):
            rhs = f'{cls._parse_rhs(rhs.signal)}[{cls._parse_rhs(rhs.index)}]'
        else:
            raise ValueError("Unknown RHS object detected: {}".format(rhs))

        return rhs

class Signal:
    def __init__(self, signal, domain = None):
        self.signal = signal
        self.domain = domain
        self.statements = []

        self._no_reset_statement = False

    def disable_reset_statement(self):
        self._no_reset_statement = True

    @property
    def assigned_bits(self):
        assigned_bits = [False] * len(self.signal)
        for statement in self.statements:
            if isinstance(statement, Assign):
                start_idx = statement._start_idx
                stop_idx = statement._stop_idx

                if start_idx is None:
                    start_idx = 0
                if stop_idx is None:
                    stop_idx = len(self.signal)

                for bit in range(start_idx, stop_idx):
                    if assigned_bits[bit]:
                        pass    # TODO: Possible Verilog error?
                    assigned_bits[bit] = True

        return assigned_bits

    @property
    def reset_statement(self):
        if self.domain is not None or self._no_reset_statement:
            return None

        assigned_bits = self.assigned_bits
        if all(assigned_bits):
            return None

        return Assign(ast.Const(self.signal.reset, len(self.signal)))

    @staticmethod
    def sanitize(name, hdl=None):
        if hdl is not None:
            name = hdl.sanitize(name)
        return name

    def add_statement(self, statement):
        if not isinstance(statement, list):
            statement = [statement]
        self.statements.extend(statement)

    @property
    def static(self):
        # TODO: This can be used to treat comb-only signals without "if" as assigns

        if self.domain is None:
            if self.statements:
                if len(self.statements) <= 1 and all(isinstance(st, Assign) for st in self.statements):
                    if all(self.assigned_bits):
                        return True
            else:
                return True

        return False

class Port(Signal):
    def __init__(self, signal, direction, domain=None):
        super().__init__(signal, domain)
        self.direction = direction

        # if self.direction == 'i':
        #     self.reset_statement = None

class Memory(Signal):
    def __init__(self, signal, init = None):
        super().__init__(signal)
        self.init = [] if init is None else init

    @property
    def reset_statement(self):
        return None

class MemoryPort(Signal):
    def __init__(self, signal, domain=None, index=None):
        super().__init__(signal, domain)
        self.index = index
        self.domain = domain

    @property
    def reset_statement(self):
        return None

class Statement:
    pass

class Assign(Statement):
    def __init__(self, rhs, start_idx=None, stop_idx=None):
        self.rhs = rhs
        self._start_idx = start_idx
        self._stop_idx = stop_idx

class Switch(Statement):

    class Case:
        def __init__(self, test, statements):
            self.test = test
            self.statements = statements

    class If(Case):
        pass

    class Else(Case):
        def __init__(self, statements):
            super().__init__(None, statements)

    def __init__(self, test, cases):
        self.test  = test
        self.cases = self.process_cases(test, cases)

        # Move default to last place
        default = self.cases.pop(None, None)
        if default is not None:
            self.cases[None] = default

        self.strip_unused_cases()

        self.as_if = self._as_if()

    def strip_unused_cases(self):
        patterns = []
        pops = []
        for case, statements in reversed(self.cases.items()):
            mask = case
            if case is None:
                mask = '?' * len(self.test)
            if not statements:
                for pattern in patterns:
                    if all(p == c or p == '?' or c == '?' for p, c in zip(pattern, mask)):
                        break
                else:
                    pops.append(case)
            else:
                patterns.append(mask)

        for pop in pops:
            self.cases.pop(pop)

    @classmethod
    def convert_case(cls, test, case):
        if isinstance(case, tuple):
            if len(case) == 0:
                case = [None]
            else:
                new_cases = []
                for c in case:
                    new_cases.extend(Switch.convert_case(test, c))
                case = new_cases
        elif isinstance(case, str):
            case = [case.replace('-', '?').replace(' ', '')]
        elif isinstance(case, int):
            case = Switch.convert_case(test, format(case, f'0{len(test)}b'))
        else:
            raise RuntimeError(f"Unknown switch case: {case}")

        return case

    @classmethod
    def process_cases(cls, test, cases):
        ret = {}

        for case, statements in cases.items():
            new_cases = cls.convert_case(test, case)

            for c in new_cases:
                ret[c] = statements

        return ret

    def _as_if(self):
        res = []

        for case, statements in self.cases.items():
            if case is None:
                res.append(self.Else(statements))
                break

            if case.count('1') != 1 or not all(c in ['?', '1'] for c in case):
                return None

            bit = case[::-1].index('1')
            res.append(self.If(self.test[bit], statements))

        return res

class Module:

    allow_remapping = True  # Not strictly necessary, but cocotb needs it to force-assign some signals

    def __init__(self, name, fragment, hdl=None, invalid_names=None, top=True, type = None):
        if invalid_names is None:
            invalid_names = set([''])

        self.invalid_names = invalid_names
        self.top = top

        self.name = name
        self.type = name if type is None else type
        self._fragment = fragment
        if top:
            self.fragment = None
        else:
            self.fragment = fragment

        self.hdl = hdl

        self.submodules = []
        self._signals = ast.SignalDict()
        self.ports = []
        self._remapped = ast.SignalDict()

    @property
    def empty(self):
        if not isinstance(self.fragment, ir.Fragment):
            return True

        res = True

        res = res and not self.fragment.ports
        res = res and not self.fragment.drivers
        res = res and not self.fragment.statements
        # res = res and not self.fragment.domains
        res = res and not self.fragment.subfragments
        res = res and not self.fragment.attrs
        res = res and not self.fragment.generated

        return res

    @property
    def domains(self):
        return self.fragment.domains

    @property
    def parameters(self):
        return {}

    @property
    def case_sensitive(self):
        return bool(getattr(self.hdl, 'case_sensitive', False))

    def _change_case(self, name):
        return name if self.case_sensitive else name.lower()

    def _sanitize(self, name, extra=None):
        # invalid = set([''])
        # invalid.update(self._change_case(submodule.name) for submodule, _ in self.submodules)
        # invalid.update(self._change_case(signal.name) for signal in self._signals)
        invalid = self.invalid_names.copy()
        if extra is not None:
            invalid.update(extra)
        if not name:
            name = 'unnamed'

        curr_num = ''
        curr_idx = len(name) - 1
        while curr_idx >= 0 and name[curr_idx].isnumeric():
            curr_num = name[curr_idx] + curr_num
            curr_idx -= 1

        if curr_num:
            idx = int(curr_num) + 1
            if self._change_case(name) in invalid:
                name = f'{name[:curr_idx+1]}{idx}'
                idx += 1
        else:
            idx = 0

        _name = name
        while self._change_case(name) in invalid:
            name = f'{_name}{idx}'
            idx += 1

        return name

    def sanitize_module(self, name, extra=None):
        if self.hdl is not None:
            name = self.hdl.sanitize(name)
        return self._sanitize(name, extra=extra)

    def _sanitize_signal(self, signal, extra=None):
        signal.name = self._sanitize(Signal.sanitize(signal.name, hdl=self.hdl), extra=extra)

    def _reset(self):
        self._signals.clear()
        self.submodules.clear()
        self.ports.clear()
        self._remapped.clear()
        if self.top:
            self.invalid_names.clear()

    def _cleanup_signal_names(self):
        extra = set()

        # For toplevel, ports have priority
        for port in self.ports:
            if self.top:
                self._sanitize_signal(port.signal, extra=extra)
            extra.add(self._change_case(port.signal.name))

        for submodule, _ in self.submodules:
            submodule.name = self.sanitize_module(submodule.name, extra=extra)
            extra.add(self._change_case(submodule.name))
            if not isinstance(submodule, InstanceModule):
                submodule.type = self.sanitize_module(submodule.type, extra=extra)
            self.invalid_names.add(self._change_case(submodule.type))

        for signal, mapping in self._signals.items():
            if not isinstance(mapping, Port):
                self._sanitize_signal(signal, extra=extra)
            extra.add(self._change_case(signal.name))

        for submodule, _ in self.submodules:
            submodule._cleanup_signal_names()

    def _get_signal(self, signal):
        s = self._signals.get(signal, None)
        if s is None:
            raise RuntimeError(f"Missing signal {signal.name} from module {self.name}")
        return s

    def _new_signal(self, shape=1, prefix=None, mapping=Signal, **kwargs):
        name = prefix or 'tmp'
        new = ast.Signal(shape, name=name)
        # self._sanitize_signal(new)
        self._signals[new] = mapping(new, **kwargs)

        return new

    def _zero_size_signal(self):
        # TODO: Check
        return ast.Const(0, 0)
        return self._new_signal(0, prefix = 'empty')

    def _add_new_statement(self, left, statement):
        ########################
        if left not in self._signals:
            # self._sanitize_signal(left)
            self._signals[left] = Signal(left)
        ########################
        remap = self._remapped.get(left, None)
        if remap is not None:
            left = remap
        self._get_signal(left).add_statement(statement)

    def _add_new_assign(self, left, right, start_idx=None, stop_idx=None):
        self._add_new_statement(left, Assign(right, start_idx, stop_idx))

    def prepare(self, ports=None, platform=None):
        if self.top:
            self.fragment = ir.Fragment.get(self._fragment, platform).prepare(ports)
            self.name = self.sanitize_module(self.name)
            self.invalid_names.add(self._change_case(self.name))

        self._prepare_signals()
        self._prepare_statements()
        self._prepare_submodules()

        if self.top:
            self._cleanup_signal_names()

    def _prepare_signals(self):

        # TODO: Possibly create intermediate signals so that ports are always wire
        for port, direction in self.fragment.ports.items():
            # self._sanitize_signal(port)
            self._signals[port] = Port(port, direction=direction)
            self.ports.append(self._signals[port])

        for domain, signal in self.fragment.iter_drivers():
            entry = self._signals.get(signal, None)
            if entry is None:
                # self._sanitize_signal(signal)
                entry = self._signals[signal] = Signal(signal)

            if self.allow_remapping and domain is not None:
                remap = self._remapped[signal] = self._new_signal(signal.shape(), prefix = f'{signal.name}_next')

                # For some reason, cocotb needs rst duplication in comb and sync parts
                if self.domains[domain].rst is None or not self.domains[domain].async_reset:
                    entry.add_statement(Assign(remap))
                else:
                    entry.add_statement(Switch(self.domains[domain].rst, {
                        '0': [Assign(remap)],
                        '1': [Assign(ast.Const(signal.reset, len(signal)))]
                    }))

                self._signals[remap].add_statement(Assign(signal)) # TODO: Avoid duplicates

            entry.domain = domain

    @staticmethod
    def _slice_check_const(rhs, start, stop):
        if isinstance(rhs, ast.Const):
            value = rhs.value
            width = min(rhs.width, stop) - start
            # signed = rhs.signed
            if rhs.value < 0:
                value += 2**rhs.width
            value = (value >> start) & int('1' * (stop - start), 2)
            # if signed:
            #     value -= 2**width

            return ast.Const(value, width)

        else:
            return rhs[start : stop]

    def _process_lhs(self, lhs, rhs, start_idx=None, stop_idx=None):
        res = []

        # TODO: Review start_idx/stop_idx, maybe it's better to use intermediate signals!

        if start_idx is None or start_idx < 0:
            start_idx = 0
        if stop_idx is None or stop_idx > len(lhs):
            stop_idx = len(lhs)

        if isinstance(lhs, (ast.Const, ast.Operator)):
            raise RuntimeError("Invalid LHS:", lhs)
        elif isinstance(lhs, ast.Signal):
            res.append((lhs, Assign(rhs, start_idx, stop_idx)))
        elif isinstance(lhs, ast.Cat):
            loffset = roffset = 0
            parts = [part for part in lhs.parts if len(part)]

            for part in parts:
                new_start = 0
                new_stop = len(part)

                if loffset + len(part) <= start_idx:
                    loffset += len(part)
                    continue

                if loffset < start_idx:
                    new_start = start_idx - loffset

                if loffset + len(part) > stop_idx:
                    new_stop = stop_idx - loffset

                if roffset >= len(rhs):
                    break

                if roffset == 0 and len(part) >= len(rhs):
                    new_rhs = rhs
                elif isinstance(rhs, ast.Slice):
                    # FIX: Avoid duplicating slices!
                    new_rhs = ast.Slice(rhs.value, rhs.start + roffset, min(rhs.start + roffset + len(part), rhs.stop))
                else:
                    new_rhs = self._slice_check_const(rhs, roffset, roffset + len(part))

                res.extend(self._process_lhs(part, new_rhs, new_start, new_stop))
                loffset += len(part)
                roffset += len(part)

        elif isinstance(lhs, ast.Slice):
            if lhs.start < lhs.stop:
                start = start_idx+lhs.start
                res.extend(self._process_lhs(lhs.value, rhs, start_idx=start, stop_idx=min(start+stop_idx, lhs.stop)))

        elif isinstance(lhs, ast.Part):
            if isinstance(lhs.offset, ast.Const):
                raise RuntimeError("Part with const offset is Slice!")
            else:
                case = 0
                cases = {}

                while True:
                    offset = case * lhs.stride
                    if offset >= len(lhs.value):
                        break

                    part = lhs.value[offset : offset + lhs.width]
                    cases[case] = self._process_lhs(part, rhs, start_idx, stop_idx)
                    case += 1

                res.extend(self._open_switch(lhs.offset, cases))

        elif isinstance(lhs, ast.ArrayProxy):
            case = 0
            cases = {}

            while True:
                idx = case
                if idx >= len(lhs.elems):
                    break

                part = lhs.elems[idx]
                cases[case] = self._process_lhs(part, rhs, start_idx, stop_idx)
                case += 1

            res.extend(self._open_switch(lhs.index, cases))

        else:
            raise ValueError("Unknown RHS object detected: {}".format(rhs.__class__.__name__))

        return res

    def _process_rhs(self, rhs, **kwargs):
        # TODO: Possibly check if return value differs input value to determine whether a new signal is needed
        # so we can reduce code size

        # TODO: Possibly receive a "top" parameter, so that the first layer doesn't need to create a new signal
        # For example: assign x = Cat(Slice, Part) ---> x = Cat(new_slice, new_part) instead of x = new_signal_for_cat

        _division_fix = kwargs.get('_division_fix', False)

        if isinstance(rhs, ast.Const):
            pass
        elif isinstance(rhs, ast.Signal):
            # Fix: Can happen with submodule ports
            if rhs not in self._signals:
                # self._sanitize_signal(rhs)
                self._signals[rhs] = Signal(rhs)

        elif isinstance(rhs, ast.Cat):
            parts = [part for part in rhs.parts if len(part)]
            if len(parts) == 0:
                rhs = self._zero_size_signal()
            elif len(parts) == 1:
                rhs = self._process_rhs(parts[0], **kwargs)
            else:
                new_rhs = self._new_signal(rhs.shape(), prefix='concat')
                rhs.parts = [self._process_rhs(part, **kwargs) for part in parts if len(part)]

                self._add_new_assign(new_rhs, rhs)
                rhs = new_rhs

        elif isinstance(rhs, ast.Slice):
            if rhs.start >= rhs.stop:
                rhs = self._zero_size_signal()
            else:
                rhs.value = self._process_rhs(rhs.value, **kwargs)

                if rhs.start == 0 and rhs.stop >= len(rhs.value):
                    rhs = rhs.value
                elif isinstance(rhs.value, ast.Const):
                    rhs = self._slice_check_const(rhs.value, rhs.start, rhs.stop)
                else:
                    new_rhs = self._new_signal(rhs.shape(), prefix='slice')
                    self._add_new_assign(new_rhs, rhs)
                    rhs = new_rhs

        elif isinstance(rhs, ast.Operator):
            for i, operand in enumerate(rhs.operands):
                rhs.operands[i] = self._process_rhs(operand, **kwargs)

            if not _division_fix and rhs.operator == '//' and len(rhs.operands) == 2:
                dividend, divisor = self._signed_division_fix(rhs)
                kwargs['_division_fix'] = True
                new_rhs = self._process_rhs(ast.Mux(divisor == ast.Const(0, len(divisor)), ast.Const(0, len(rhs)), dividend//divisor), **kwargs)

            else:
                new_rhs = self._new_signal(rhs.shape(), prefix='operand')
                self._add_new_assign(new_rhs, rhs)

            rhs = new_rhs

        elif isinstance(rhs, ast.Part):
            if isinstance(rhs.offset, ast.Const):
                raise RuntimeError("Part with const offset is Slice!")
            else:
                shift = rhs.offset

                if rhs.stride > 1:
                    shift = shift * rhs.stride

                rhs = self._process_rhs((rhs.value >> shift)[:rhs.width], **kwargs)   # TODO: This is cleaner, but may cause issues!

        elif isinstance(rhs, ast.ArrayProxy):
            if not rhs.elems:
                rhs = self._zero_size_signal()
            else:
                for i, elem in enumerate(rhs.elems):
                    rhs.elems[i] = self._process_rhs(elem, **kwargs)

                rhs.index = self._process_rhs(rhs.index, **kwargs)

                new_rhs = self._new_signal(rhs.shape(), prefix='array')

                index = self._process_rhs(rhs.index, **kwargs)
                cases = {
                    i: [Assign(elem)] for i, elem in enumerate(rhs.elems)
                }

                # if 2**len(index) > len(rhs.elems):
                #     cases[None] = 0 # Default

                self._add_new_statement(new_rhs, Switch(index, cases))
                rhs = new_rhs
        else:
            raise ValueError("Unknown RHS object detected: {}".format(rhs.__class__.__name__))

        return rhs

    def _signed_division_fix(self, rhs):
        dividend, divisor = rhs.operands

        if any(operand.shape().signed for operand in rhs.operands):
            max_size = max(len(op) for op in rhs.operands) + 2

            if not dividend.shape().signed:
                dividend = dividend.as_signed()

            if not divisor.shape().signed:
                divisor = divisor.as_signed()

            dividend = self._fix_rhs_size(dividend, max_size)
            divisor = self._fix_rhs_size(divisor, max_size)

            # Yosys fix for signed division
            dividend = self._process_rhs(ast.Mux(
                (dividend[-1] == divisor[-1]) | (dividend == ast.Const(0, len(dividend))),
                dividend,
                dividend - ast.Mux(divisor[-1], divisor + ast.Const(1, len(divisor)), divisor - ast.Const(1, len(divisor)))
            ))

        return dividend, divisor

    def _fix_rhs_size(self, rhs, size=None, *, _force_sign=None, _allow_upsize=False):
        if size is None:
            size = len(rhs)

        # Fix: Zero-width signals!
        if len(rhs) == 0 and size > 0:
            rhs = ast.Const(0, size)

        if isinstance(rhs, ast.Const):

            signed = rhs.signed
            new_value = rhs.value

            # Downsize
            ################################
            if rhs.width > size and not _allow_upsize:
                if signed:
                    new_value += 2**rhs.width
                new_value = new_value & ((1 << size) - 1)
                signed = False
            ################################

            if _force_sign is not None:
                if _allow_upsize and _force_sign and not signed and (new_value >> (size - 1)):
                    # Overflow!
                    # TODO: Maybe check sign bit before upsizing?
                    size += 1
                signed = _force_sign

            rhs = ast.Const(new_value, ast.Shape(size, signed))

        else:
            signed = rhs.shape().signed
            if _force_sign is not None and _force_sign != signed:
                # TODO: Always necessary or just for unsigned->signed?
                if _allow_upsize:
                    size += 1

                new_rhs = self._new_signal(ast.Shape(size, _force_sign), prefix = 'resigned')
                self._add_new_assign(new_rhs, self._fix_rhs_size(rhs, size))
                return new_rhs

            if isinstance(rhs, ast.Cat):
                for i, part in enumerate(rhs.parts):
                    rhs.parts[i] = self._fix_rhs_size(part)
            elif isinstance(rhs, ast.Slice):
                rhs.value = self._fix_rhs_size(rhs.value)
            elif isinstance(rhs, ast.Part):
                rhs.offset = self._fix_rhs_size(rhs.offset)
                rhs.value = self._fix_rhs_size(rhs.value)
            elif isinstance(rhs, ast.ArrayProxy):
                rhs.index = self._fix_rhs_size(rhs.index)
                for i, elem in enumerate(rhs.elems):
                    rhs.elems[i] = self._fix_rhs_size(elem)

            if isinstance(rhs, (ast.Signal, ast.Cat, ast.Slice, ast.Part, ast.ArrayProxy)):
                if len(rhs) < size or (len(rhs) > size and not _allow_upsize):
                    new_rhs = self._new_signal(ast.Shape(size, signed=rhs.shape().signed), prefix = 'resized')
                    self._add_new_assign(new_rhs, self._fix_rhs_size(rhs))
                    rhs = new_rhs

            elif isinstance(rhs, ast.Operator):
                operands = rhs.operands
                max_size = max(size, max(len(op) for op in operands))

                if len(operands) == 1:
                    # TODO: Do we need to force sign?
                    if rhs.operator == 'u':
                        signed = False
                    elif rhs.operator == 's':
                        signed = True
                    else:
                        signed = _force_sign
                    rhs.operands = [self._fix_rhs_size(operands[0], max_size, _force_sign=signed)]

                elif len(operands) == 2:
                    if rhs.operator in ('<<', '>>'):
                        max_size = max(size, len(operands[0]))
                        rhs.operands = [
                            self._fix_rhs_size(operands[0], max_size),
                            self._fix_rhs_size(operands[1], _force_sign=False),
                        ]


                        # TODO: Size change possibility
                        # rhs.operands[1] = self._fix_rhs_size(rhs.operands[1], _force_sign=False)

                        # if isinstance(rhs.operands[1], ast.Const):
                        #     shift = rhs.operands[1].value
                        # else:
                        #     shift = 2**len(rhs.operands[1]) - 1

                        # if rhs.operator == '<<':
                        #     new_size = max(len(rhs.operands[0]), size - shift)
                        # else:
                        #     new_size = max(len(rhs.operands[0]), size + shift)

                        # rhs.operands[0] = self._fix_rhs_size(rhs.operands[0], new_size)

                        # FIX: Ignore upsize constraint, need to force shift size early to avoid infinitely large signals
                        new_rhs = self._new_signal(size, prefix = 'shifted')
                        self._add_new_assign(new_rhs, rhs)
                        rhs = new_rhs
                    else:
                        signed = any(op.shape().signed for op in operands)
                        for i, operand in enumerate(rhs.operands):
                            rhs.operands[i] = self._fix_rhs_size(operand, _allow_upsize=True, _force_sign=signed)

                        signed = any(op.shape().signed for op in operands)
                        max_size = max(size, max(len(op) for op in rhs.operands))
                        rhs.operands = [
                            self._fix_rhs_size(op, max_size, _force_sign=signed) for op in operands
                        ]

                elif len(operands) == 3:
                    signed = any(op.shape().signed for op in operands[1:])
                    rhs.operands = [
                        self._fix_rhs_size(operands[0]),
                        *[self._fix_rhs_size(op, max_size, _force_sign=signed) for op in operands[1:]]
                    ]
                else:
                    raise RuntimeError(f"Unknown operator and operands: {rhs.operator}, {rhs.operands}")

                if _force_sign is None:
                    signed = rhs.shape().signed
                else:
                    signed = _force_sign

                if len(rhs) < size or (len(rhs) > size and not _allow_upsize):
                    new_rhs = self._new_signal(ast.Shape(size, signed=signed), prefix = 'expanded_op')
                    self._add_new_assign(new_rhs, rhs)
                    rhs = new_rhs

            else:
                raise ValueError("Unknown RHS object detected: {}".format(rhs.__class__.__name__))

        return self._process_rhs(rhs)

    def _open_switch(self, test, cases):
        res = []
        per_signal = ast.SignalDict()

        all_signals = ast.SignalSet()
        for case, statements in cases.items():
            for signal, st in statements:
                all_signals.add(signal)

        for case, statements in cases.items():
            for signal in all_signals:
                per_signal.setdefault(signal, {}).setdefault(case, [])

            for signal, st in statements:
                per_signal[signal][case].append(st)

        test = self._fix_rhs_size(test)

        res = []
        for signal, cases in per_signal.items():
            res.append((signal, Switch(test, cases)))

        return res

    def _process_assign(self, assign: ast.Assign):
        # TODO: Is upsize allowed here?
        return self._process_lhs(assign.lhs, self._fix_rhs_size(assign.rhs, len(assign.lhs), _allow_upsize=True))

    def _process_switch(self, switch: ast.Switch):
        cases = {}
        for case, statements in switch.cases.items():
            for statement in statements:
                cases.setdefault(case, []).extend(self._process_statement(statement))

        return self._open_switch(switch.test, cases)

    def _process_statement(self, statement):
        res = []
        if isinstance(statement, ast.Assign):
            res.extend(self._process_assign(statement))

        elif isinstance(statement, ast.Switch):
            res.extend(self._process_switch(statement))

        elif isinstance(statement, ast._StatementList):
            for st in statement:
                res.extend(self._process_statement(st))
        else:
            raise ValueError("Unknown statement: {}".format(statement.__class__.__name__))

        return res

    def _execute_statements(self, statements):
        for statement in statements:
            for signal, st in self._process_statement(statement):
                self._add_new_statement(signal, st)

    def _prepare_statements(self):
        self._execute_statements(self.fragment.statements)

    def _submodule_create(self, name, fragment, cls=None, **kwargs):
        if cls is None:
            cls = Module

        return cls(name, fragment, hdl=self.hdl, invalid_names=self.invalid_names, top=False, **kwargs)

    def _process_memory(self, subfragment, name):
        m = self._submodule_create(name, subfragment, MemoryModule)
        m.prepare()

        for signal, mapping in m._signals.items():

            if isinstance(mapping, (Memory, MemoryPort)):
                assert signal not in self._signals, "Internal error with Memory signals"
                self._signals[signal] = mapping

            else:
                curr_mapping = self._signals.get(signal, None)
                if curr_mapping is None:
                    self._signals[signal] = curr_mapping = Signal(mapping.signal, mapping.domain)

                if curr_mapping.domain is None:
                    curr_mapping.domain = mapping.domain    # Allow for domain changes

                for statement in mapping.statements:
                    self._signals[signal].add_statement(statement)

    def _process_submodule_instance(self, subfragment, name):
        ports = {}

        if subfragment.type == "$mem_v2":   # TODO: Check if there's a better way to determine this
            self._process_memory(subfragment, name)
            submodule = None
        else:
            submodule = self._submodule_create(name, subfragment, InstanceModule)
            submodule.prepare()
            for port_name, (port_value, kind) in subfragment.named_ports.items():
                local_signal = None
                if kind == 'io':
                    ports[port_name] = port_value   # TODO: Check how to handle!
                    local_signal = self._signals.get(port_value, None)
                else:
                    # Special case for ports tied to constants/signals -- We can reduce some logic
                    if kind == 'i' and (
                        isinstance(port_value, ast.Const) or
                        (isinstance(port_value, ast.Signal) and port_value in self._signals)
                    ):
                        ports[port_name] = port_value
                    else:
                        new_port = self._new_signal(port_value.shape(), prefix=f'port_{port_name}')
                        ports[port_name] = new_port
                        if kind == 'i':
                            self._execute_statements([new_port.eq(port_value)])
                        elif kind == 'o':
                            self._execute_statements([port_value.eq(new_port)])
                            local_signal = self._signals[new_port]
                        else:
                            raise RuntimeError(f"Unknown port type for port {port_name} for submodule {submodule.name} of module {self.name}: {kind}")

                if local_signal is not None:
                    local_signal.disable_reset_statement()

        return submodule, ports

    def _prepare_submodules(self):
        for subfragment, subname in self.fragment.subfragments:
            if subname is None:
                subname = 'unnamed'

            if isinstance(subfragment, ir.Instance):
                submodule, ports = self._process_submodule_instance(subfragment, subname)
            else:
                submodule = self._submodule_create(subname, subfragment, type = f'{self.type}_{subname}')
                submodule.prepare()
                ports = None
                for port in submodule.ports:
                    local_signal = self._signals.get(port.signal, None)
                    if local_signal is None:
                        local_signal = self._signals[port.signal] = Signal(port.signal)

                    if port.direction in ['o', 'io']:
                        local_signal.disable_reset_statement()

            if submodule is not None:
                self.submodules.append((submodule, ports))

class InstanceModule(Module):
    def __init__(self, name, fragment, hdl=None, invalid_names=None, top=True):
        super().__init__(name, fragment, hdl=hdl, invalid_names=invalid_names, top=top, type=fragment.type)

    @property
    def parameters(self):
        return self.fragment.parameters

class MemoryModule(InstanceModule):

    allow_remapping = False

    class Port:
        def __init__(self, data, index, enable, domain):
            self.data = data
            self.index = index
            self.enable = enable
            self.domain = domain

            self.proxy = None

        @staticmethod
        def _get_signal(name, n, ports):
            signal = ports.get(name, None)
            if name is None:
                raise RuntimeError(f"Failed to find signal {name} for memory port")
            signal = signal[0]
            assert isinstance(signal, ast.Cat), f"Port {name} for memory port must be a concatenation"
            return signal.parts

        @classmethod
        def from_fragment(cls, fragment, prefix, domain_resolver):
            ports = fragment.named_ports
            parameters = fragment.parameters

            n = parameters.get(f'{prefix}_PORTS', 0)

            datas = cls._get_signal(f'{prefix}_DATA', n, ports)
            indexes = cls._get_signal(f'{prefix}_ADDR', n, ports)
            enables = cls._get_signal(f'{prefix}_EN', n, ports)

            clk = cls._get_signal(f'{prefix}_CLK', n, ports)
            domains = []

            for i in range(n):
                if isinstance(clk[i], ast.Const) and clk[i].value == 0:
                    domain = None
                else:
                    domain = domain_resolver(clk[i])

                domains.append(domain)

            return [
                cls(*entry) for entry in zip(datas, indexes, enables, domains)
            ]

    class ReadPort(Port):
        @classmethod
        def from_fragment(cls, fragment, domain_resolver):
            return super().from_fragment(fragment, 'RD', domain_resolver)

    class WritePort(Port):
        @classmethod
        def from_fragment(cls, fragment, domain_resolver):
            return super().from_fragment(fragment, 'WR', domain_resolver)

    def __init__(self, name, fragment, hdl=None, invalid_names=None, top=True):
        super().__init__(name, fragment, hdl=hdl, invalid_names=invalid_names, top=top)

        self._mem = None
        self._arr = ast.SignalSet()

        self._r_ports = self.ReadPort.from_fragment(fragment, domain_resolver=self._find_domain_from_clock)
        self._w_ports = self.WritePort.from_fragment(fragment, domain_resolver=self._find_domain_from_clock)

        self._width = fragment.parameters.get('WIDTH', None)
        if self._width is None:
            raise RuntimeError(f"Failed to find width for memory {self.name}")

        self._size = fragment.parameters.get('SIZE', None)
        if self._size is None:
            raise RuntimeError(f"Failed to find size for memory {self.name}")

        init = fragment.parameters.get('INIT', ast.Const(0, self._width * self._size)).value
        self._init = [
            ast.Const((init >> (self._width * i)) & int('1' * self._width, 2), self._width) for i in range(self._size)
        ]

    def _find_domain_from_clock(self, clock):
        for domain in self.domains.values():
            if domain.clk is clock:
                return domain
        raise RuntimeError(f"Failed to find domain for clock {clock} of {self.name}")

    def _prepare_signals(self):
        super()._prepare_signals()

        # TODO: Handle granularity!!!

        self._mem = self._signals[self._new_signal(
            shape   = self._width,
            prefix  = self.name,
            mapping = Memory,
            init    = self._init,
        )]

        for rport in self._r_ports:
            rdomain = rport.domain.name if rport.domain is not None else None
            self._signals[rport.data].domain = rdomain

            # TODO: Avoid using combinational proxy and directly use mem[addr]
            rport.proxy = self._new_signal(shape = self._width, prefix = f'{self.name}_r_data')

            if rport.domain is None:  # TODO: Also here if r_en is never assigned by parent module
                self._signals.pop(rport.enable, None)

            self._add_new_statement(rport.proxy, Assign(MemoryPort(self._mem.signal, domain = rdomain, index = rport.index)))

        for wport in self._w_ports:
            wdomain = wport.domain.name if wport.domain is not None else None
            wport.proxy = self._new_signal(
                shape = self._width,
                prefix = f'{self.name}_internal',    # Doesn't matter, signal won't really exist
                mapping = MemoryPort,
                domain = wdomain,
                index = wport.index,
            )
            self._signals[wport.proxy].signal = self._mem.signal    # Hacky!

        self._update_statements(self.fragment.statements)

    def _reset(self):
        super()._reset()
        self._arr.clear()

    def _set_arr(self, arr):
        for signal in arr.elems:
            self._signals.pop(signal, None)

        self._arr.update(arr.elems)

    def _process_rhs(self, rhs, **kwargs):
        if isinstance(rhs, ast.ArrayProxy):
            self._set_arr(rhs)

            for rport in self._r_ports:
                if rhs.index is rport.index:
                    return rport.proxy

            raise RuntimeError(f"Port read index not found for memory {self.name}")

        return super()._process_rhs(rhs, **kwargs)

    def _process_lhs(self, lhs, rhs, start_idx=None, stop_idx=None):
        if isinstance(lhs, ast.ArrayProxy):
            self._set_arr(lhs)

            for wport in self._w_ports:
                if lhs.index is wport.index:
                    return [(wport.proxy, Assign(rhs))]

            raise RuntimeError(f"Port write index not found for memory {self.name}")

        elif isinstance(lhs, ast.Signal) and self._arr is not None and lhs in self._arr:
            return []

        # else:
        #     for rport in self._r_ports:
        #         if lhs is rport.data:
        #             return [(rport.proxy, Assign(rhs))]

        return super()._process_lhs(lhs, rhs, start_idx, stop_idx)

    def _update_statements(self, statements):
        replace_statements = {}

        for i, st in enumerate(statements):
            if not isinstance(st, ast.Switch):
                continue

            if any(
                st.test is rport.enable and rport.domain is None for rport in self._r_ports
            ):
                assert [('1',)] == list(st.cases), f"Invalid memory statement for module {self.name}"
                replace_statements[i] = ast._StatementList(st.cases[('1',)])

            else:
                rst_cond = any(
                    rport.domain is not None and st.test is rport.domain.rst for rport in self._r_ports
                )
                rst_cond = rst_cond or any(
                    wport.domain is not None and st.test is wport.domain.rst for wport in self._r_ports
                )
                if rst_cond:
                    replace_statements[i] = None

        for i, new in reversed(replace_statements.items()):
            if new is None:
                statements.pop(i)
            else:
                statements[i] = new