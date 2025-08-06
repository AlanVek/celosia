from amaranth.hdl import ast, ir
from textwrap import dedent, indent

# TODO: If we find multiple signals with same statements, maybe we can merge them into one!

class HDL:
    case_sensitive = False

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

class VHDL(HDL):
    case_sensitive = True

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

        return name

    @classmethod
    def _convert_module(cls, module):
        pass

class Verilog(HDL):
    case_sensitive = False
    protected = [
        'always',         'and',            'assign',         'begin',
        'buf',            'bufif0',         'bufif1',         'case',
        'casex',          'casez',          'cmos',           'deassign',
        'default',        'defparam',       'design',         'disable',
        'edge',           'else',           'end',            'endcase',
        'endfunction',    'endmodule',      'endprimitive',   'endspecify',
        'endtable',       'endtask',        'event',          'for',
        'force',          'forever',        'fork',           'function',
        'highz0',         'highz1',         'if',             'ifnone',
        'initial',        'inout',          'input',          'integer',
        'join',           'large',          'localparam',     'macromodule',
        'medium',         'module',         'nand',           'negedge',
        'nmos',           'nor',            'not',            'notif0',
        'notif1',         'or',             'output',         'parameter',
        'pmos',           'posedge',        'primitive',      'pull0',
        'pull1',          'pulldown',       'pullup',         'rcmos',
        'real',           'realtime',       'reg',            'release',
        'repeat',         'rnmos',          'rpmos',          'rtran',
        'rtranif0',       'rtranif1',       'scalared',       'small',
        'specify',        'specparam',      'strong0',        'strong1',
        'supply0',        'supply1',        'table',          'task',
        'time',           'tran',           'tranif0',        'tranif1',
        'tri',            'tri0',           'tri1',           'triand',
        'trior',          'trireg',         'vectored',       'wait',
        'wand',           'weak0',          'weak1',          'while',
        'wire',           'wor',            'xnor',           'xor',
    ]

    template = """module {name} (
{port_block}
);
{initials_block}{submodules_block}{blocks_block}{assignment_block}
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
        }

        for old, new in replace_map.items():
            name = name.replace(old, new)

        while name in cls.protected:
            name = 'esc_' + name

        if not name:
            name = cls.sanitize('unnamed')

        return name

    @classmethod
    def _convert_module(cls, module):
        if module.empty:
            return ''

        port_block, initial_block, assignment_block, blocks_block = cls._parse_signals(module)
        submodules_block = cls._generate_submodule_blocks(module)

        # TODO: Guarantee no collisions with Instance names

        res = cls.template.format(
            name = module.name,
            port_block = port_block,
            initials_block = initial_block,
            assignment_block = assignment_block,
            submodules_block = submodules_block,
            blocks_block = blocks_block,
        )
        for submodule, ports in module.submodules:
            if ports is not None:   # Instance
                continue
            res += '\n' + cls._convert_module(submodule)

        return res

    @classmethod
    def _parse_signals(cls, module):
        port_block = initial_block = assignment_block = blocks_block = ''

        for mapping in module._signals.values():
            initial_block += cls._generate_initial(mapping)
            if isinstance(mapping, Port):
                port_block += f'{cls._generate_one_port(mapping)},\n'

                if mapping.direction == 'i':
                    continue

            if mapping.static:
                if mapping.statements:
                    statement = mapping.statements[0]
                elif mapping.reset_statement is not None:
                    statement = mapping.reset_statement
                else:
                    continue
                assignment_block += f'assign {cls._generate_one_assignment(mapping, statement, symbol="=")};\n'
            else:
                blocks_block += f'{cls._generate_one_block(mapping, module)}'

        port_block = port_block[:-2]
        blocks = (port_block, initial_block, assignment_block, blocks_block)
        return (indent(block, ' '*4) for block in blocks)

    @classmethod
    def _generate_initial(cls, mapping):
        res = ''

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
            res += ';\n'

        if isinstance(mapping, Memory):
            res += 'initial begin\n'
            for i, reset in enumerate(mapping.init):
                res += f'    {mapping.signal.name}[{i}] = {cls._parse_rhs(reset)};\n'
            res += 'end\n'

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

        # TODO: Validate rhs size == lhs size or take slice
        repr = mapping.signal.name

        if isinstance(mapping, Memory):
            repr = f'{repr}[{cls._parse_rhs(mapping.w_index)}]'
        elif start_idx is not None and stop_idx is not None:
            if start_idx != 0 or stop_idx != len(mapping.signal):
                if stop_idx == start_idx + 1:
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
        blocks = indent(cls._generate_statements(mapping, statements, symbol=symbol)[:-1], ' '*4)
        footer = 'end'

        return dedent(f"{header}\n{blocks}\n{footer}\n")

    @classmethod
    def _generate_statements(cls, mapping, statements, symbol):
        res = ''
        for statement in statements:
            res += cls._generate_one_statement(mapping, statement, symbol)
        return res

    @classmethod
    def _generate_one_statement(cls, mapping, statement, symbol):
        res = ''

        if isinstance(statement, Assign):
            res += f"{cls._generate_one_assignment(mapping, statement, symbol=symbol)};\n"

        elif isinstance(statement, Switch):
            res += f"{cls._generate_switch(mapping, statement, symbol=symbol)}"

        else:
            raise RuntimeError(f"Unknown statement: {statement}")

        return res

    @classmethod
    def _generate_switch(cls, mapping, statement, symbol):
        header = f'casez ({cls._parse_rhs(statement.test)})'

        # TODO: Change switch with 0/1 to if

        body = ''
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
                end = '    end\n'
            else:
                begin = end = ''

            body += f'    {case}:{begin}\n'

            if statements:
                case_body = cls._generate_statements(mapping, statements, symbol=symbol)
            else:
                case_body = '/* empty */;\n'    # TODO: Filter empty cases when possible

            body += indent(case_body, ' '*8) + end

        footer = 'endcase'

        return dedent(f'{header}\n{body[:-1]}\n{footer}\n')

    @classmethod
    def _generate_submodule_blocks(cls, module):
        res = ''

        for submodule, ports in module.submodules:
            params = {}
            if ports is None:
                if isinstance(submodule.fragment, ir.Instance):
                    raise RuntimeError(f"Found invalid submodule configuration for submodule {submodule.name} of module {module.name}")

                if submodule.empty:
                    continue

                ports = {}
                for port in submodule.ports:
                    # if port.signal not in module._signals:
                    #     raise RuntimeError(f"Found port {port.signal.name} of submodule {name} which is not a signal of {module.name}")
                    ports[port.signal.name] = port.signal

                type = submodule.name
            else:
                if not isinstance(submodule.fragment, ir.Instance):
                    raise RuntimeError(f"Found invalid submodule configuration for submodule {submodule.name} of module {module.name}")

                params.update(submodule.fragment.parameters)
                type = submodule.fragment.type

            res += f'{type}'

            if params:
                res += ' #(\n'
                for key, value in params.items():
                    res += f'    .{key}({value}),\n'   # TODO: Check types
                res = res[:-2] + '\n)'

            res += f' {submodule.name} (\n'
            for key, value in ports.items():
                res += f'    .{key}({cls._parse_rhs(value)}),\n'

            res = res[:-2] + '\n);\n'

        return indent(res, ' '*4)

    @classmethod
    def _parse_rhs(cls, rhs):
        if isinstance(rhs, ast.Const):
            fmt = 'h'
            sign = ''
            if rhs.value < 0:
                sign += '-'
                fmt = f's{fmt}'
            rhs = f"{sign}{rhs.width}'{fmt}{hex(abs(rhs.value))[2:]}"
        elif isinstance(rhs, ast.Signal):
            rhs = rhs.name
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

                if isinstance(rhs.value, ast.Const):
                    value = rhs.value.value
                    if value < 0:
                        value += 2**rhs.value.width
                    value = format(value, f'0{rhs.value.width}b')[::-1][rhs.start:rhs.stop][::-1]
                    rhs = cls._parse_rhs(ast.Const(int(value, 2), len(value)))
                else:
                    rhs = f"{cls._parse_rhs(rhs.value)}[{idx}]"

        elif isinstance(rhs, ast.Operator):
            parsed = list(map(cls._parse_rhs, rhs.operands))
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
                    rhs = p0    # TODO: Check
                elif rhs.operator == "s":
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
            rhs = f'{cls._parse_rhs(rhs.value)} >> {cls._parse_rhs(rhs.offset)}'   # TODO: Check size mismatch (probably OK)
        elif isinstance(rhs, Memory):
            rhs = f'{cls._parse_rhs(rhs.signal)}[{cls._parse_rhs(rhs.r_index)}]'
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
    def reset_statement(self):
        if self.domain is not None or self._no_reset_statement:
            return None

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

        if all(assigned_bits):
            return None

        return Assign(ast.Const(0, len(self.signal)))

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

        if self.domain is None and len(self.statements) <= 1 and all(isinstance(st, Assign) for st in self.statements):
            return True

        return False

class Port(Signal):
    def __init__(self, signal, direction, domain=None):
        super().__init__(signal, domain)
        self.direction = direction

        # if self.direction == 'i':
        #     self.reset_statement = None

class Memory(Signal):
    def __init__(self, signal, domain = None, w_index=None, r_index=None, init = None):
        super().__init__(signal, domain=domain)
        self.w_index = w_index
        self.r_index = r_index
        self.init = [] if init is None else init

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
    def __init__(self, test, cases):
        self.test  = test
        self.cases = {
            self.convert_case(test, case): statements for case, statements in cases.items()
        }

        # Move default to last place
        default = self.cases.pop(None, None)
        if default is not None:
            self.cases[None] = default

    @staticmethod
    def convert_case(test, case):
        if isinstance(case, tuple):
            if len(case) == 1:
                case = Switch.convert_case(test, case[0])
            elif len(case) == 0:
                case = None
            else:
                raise NotImplementedError("Case {} not supported".format(case))
        elif isinstance(case, str):
            if '-' in case or '?' in case:
                case = case.replace('-', '?')
        elif isinstance(case, int):
            case = Switch.convert_case(test, format(case, f'0{len(test)}b'))

        return case

    def converted_as_if(self):
        real_cases = {
            case: statements for case, statements in self.cases.items() if statements
        }

        # TODO: Check covered cases with ? to determine whether all cases are covered (equivalent to using default)!

        if None in real_cases:
            # If default has statements, allow if/else uf up to three cases if(==) else if (!=)
            if len(self.cases) <= 3:
                pass

        else:
            pass

        return None

class Module:
    def __init__(self, name, fragment, hdl=None, invalid_names=None, top=True):
        if invalid_names is None:
            invalid_names = []

        self.invalid_names = invalid_names
        self.top = top

        self.name = name
        self._fragment = fragment
        if top:
            self.fragment = None
        else:
            self.fragment = fragment

        self.hdl = hdl

        self.submodules = []
        self._signals = ast.SignalDict()
        self.ports = []

        self.invalid_names.append(self._change_case(name))

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
    def case_sensitive(self):
        return bool(getattr(self.hdl, 'case_sensitive', False))

    def _change_case(self, name):
        return name if self.case_sensitive else name.lower()

    def _sanitize(self, name, extra=None):
        invalid = set(['', self.name])
        invalid.update(self._change_case(submodule.name) for submodule, _ in self.submodules)
        # invalid.update(self._change_case(signal.name) for signal in self._signals)
        invalid.update(self.invalid_names)
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
        if self.top:
            self.invalid_names.clear()

    def _cleanup_signal_names(self):
        extra = []
        for signal in self._signals:
            self._sanitize_signal(signal, extra=extra)
            extra.append(self._change_case(signal.name))

        for submodule, _ in self.submodules:
            submodule._cleanup_signal_names()

    def _get_signal(self, signal):
        s = self._signals.get(signal, None)
        if s is None:
            raise RuntimeError(f"Missing signal {signal.name} from module {self.name}")
        return s

    def _new_signal(self, width=1, prefix=None, mapping=Signal, **kwargs):
        name = prefix or 'tmp'
        new = ast.Signal(width, name=name)
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
        self._get_signal(left).add_statement(statement)

    def _add_new_assign(self, left, right, start_idx=None, stop_idx=None):
        self._add_new_statement(left, Assign(right, start_idx, stop_idx))

    def prepare(self, ports=None, platform=None):
        if self.top:
            self.fragment = ir.Fragment.get(self._fragment, platform).prepare(ports)

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
            entry.domain = domain

    def _process_lhs(self, lhs, rhs, start_idx=None, stop_idx=None):
        res = []

        if start_idx is None or start_idx < 0:
            start_idx = 0
        if stop_idx is None or stop_idx > len(lhs):
            stop_idx = len(lhs)

        if isinstance(lhs, (ast.Const, ast.Operator)):
            raise RuntimeError("Invalid LHS:", lhs)
        elif isinstance(lhs, ast.Signal):
            res.append((lhs, Assign(rhs, start_idx, stop_idx)))
        elif isinstance(lhs, ast.Cat):
            if len(lhs.parts) == 0:
                pass
            # elif len(lhs.parts) == 1:
            #     res.extend(self._process_lhs(lhs.parts[0], rhs, start_idx, stop_idx))
            else:
                offset = 0
                for part in lhs.parts:
                    new_start = 0
                    new_stop = len(part)

                    if offset + len(part) <= start_idx:
                        offset += len(part)
                        continue

                    if offset < start_idx:
                        new_start = start_idx - offset

                    if offset + len(part) > stop_idx:
                        new_stop = stop_idx - offset

                    if offset >= len(rhs):
                        break

                    if offset == 0 and len(part) >= len(rhs):
                        new_rhs = rhs
                    else:
                        new_rhs = rhs[offset : offset + len(part)]

                    res.extend(self._process_lhs(part, new_rhs, new_start, new_stop))
                    offset += len(part)

        elif isinstance(lhs, ast.Slice):
            if lhs.start < lhs.stop:
                start = start_idx+lhs.start
                res.extend(self._process_lhs(lhs.value, rhs, start_idx=start, stop_idx=start+min(stop_idx, lhs.stop)))

        elif isinstance(lhs, ast.Part):
            if isinstance(lhs.offset, ast.Const):
                raise RuntimeError("Part with const offset is Slice!")
            else:
                case = 0
                cases = {}

                while True:
                    offset = case * lhs.stride
                    if offset >= len(lhs):
                        break

                    part = lhs.value[offset : offset + lhs.stride]
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

    def _process_rhs(self, rhs):
        if isinstance(rhs, ast.Const):
            pass
        elif isinstance(rhs, ast.Signal):
            # Fix: Can happen with submodule ports
            if rhs not in self._signals:
                # self._sanitize_signal(rhs)
                self._signals[rhs] = Signal(rhs)
        elif isinstance(rhs, ast.Cat):
            if len(rhs.parts) == 0:
                rhs = self._zero_size_signal()
            elif len(rhs.parts) == 1:
                rhs = self._process_rhs(rhs.parts[0])
            else:
                new_rhs = self._new_signal(len(rhs), prefix='concat')
                for i, part in enumerate(rhs.parts):
                    rhs.parts[i] = self._process_rhs(part)

                self._add_new_assign(new_rhs, rhs)
                rhs = new_rhs

        elif isinstance(rhs, ast.Slice):
            if rhs.start >= rhs.stop:
                rhs = self._zero_size_signal()
            else:
                rhs.value = self._process_rhs(rhs.value)

                if rhs.start == 0 and rhs.stop >= len(rhs.value):
                    rhs = rhs.value
                else:
                    new_rhs = self._new_signal(len(rhs), prefix='slice')
                    self._add_new_assign(new_rhs, rhs)
                    rhs = new_rhs

        elif isinstance(rhs, ast.Operator):
            new_rhs = self._new_signal(len(rhs), prefix='operand')
            for i, operand in enumerate(rhs.operands):
                rhs.operands[i] = self._process_rhs(operand)

            self._add_new_assign(new_rhs, rhs)
            rhs = new_rhs

        elif isinstance(rhs, ast.Part):
            if isinstance(rhs.offset, ast.Const):
                raise RuntimeError("Part with const offset is Slice!")
            else:
                new_rhs = self._new_signal(rhs.stride, prefix='part')
                rhs.value = self._process_rhs(rhs.value)
                rhs.offset = self._process_rhs(rhs.offset)

                self._add_new_assign(new_rhs, rhs)
                rhs = new_rhs

        elif isinstance(rhs, ast.ArrayProxy):
            if not rhs.elems:
                rhs = self._zero_size_signal()
            else:
                for i, elem in enumerate(rhs.elems):
                    rhs.elems[i] = self._process_rhs(elem)

                rhs.index = self._process_rhs(rhs.index)

                new_rhs = self._new_signal(max(len(elem) for elem in rhs.elems), prefix='array')

                index = self._process_rhs(rhs.index)
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

    def _open_switch(self, test, cases):
        res = []

        per_signal = ast.SignalDict()
        for case, statements in cases.items():
            for signal, st in statements:
                per_signal.setdefault(signal, {}).setdefault(case, []).append(st)

        for signal_cases in per_signal.values():
            for case in cases:
                signal_cases.setdefault(case, [])

        res = []
        for signal, cases in per_signal.items():
            res.append((signal, Switch(self._process_rhs(test), cases)))

        return res

    def _process_assign(self, assign: ast.Assign):
        return self._process_lhs(assign.lhs, self._process_rhs(assign.rhs))

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

    def _process_memory(self, submodule):
        fragment = submodule.fragment

        m = MemoryModule(submodule.name, fragment, hdl=self.hdl, invalid_names=self.invalid_names, top=False)
        m.prepare()

        for signal, mapping in m._signals.items():

            # TODO: Make it so that the memory is read only once using Amaranth's __0__
            if mapping is m._mem:
                # self._sanitize_signal(signal)
                self._signals[signal] = mapping
            else:
                if signal not in self._signals:
                    # self._sanitize_signal(signal)
                    self._signals[signal] = Signal(mapping.signal, mapping.domain)
                for statement in mapping.statements:
                    self._signals[signal].add_statement(statement)

    def _process_submodule_instance(self, submodule):
        ports = {}

        subfragment = submodule.fragment

        if subfragment.type == "$mem_v2":   # TODO: Check if there's a better way to determine this
            self._process_memory(submodule)
            submodule = None
        else:
            submodule.prepare()
            for port_name, (port_value, kind) in subfragment.named_ports.items():
                local_signal = None
                if kind == 'io':
                    ports[port_name] = port_value   # TODO: Check how to handle!
                    local_signal = self._signals.get(port_value, None)
                else:
                    new_port = self._new_signal(len(port_value), prefix=f'port_{port_name}')
                    ports[port_name] = new_port
                    if kind == 'i':
                        self._execute_statements([new_port.eq(port_value)])
                    elif kind == 'o':
                        self._execute_statements([port_value.eq(new_port)])
                    else:
                        raise RuntimeError(f"Unknown port type for port {port_name} for submodule {submodule.name} of module {self.name}: {kind}")

                if local_signal is not None:
                    local_signal.disable_reset_statement()

        return submodule, ports

    def _prepare_submodules(self):
        for subfragment, subname in self.fragment.subfragments:
            if subname is None:
                subname = 'unnamed'

            name = self.sanitize_module(subname)
            submodule = Module(name, subfragment, hdl=self.hdl, invalid_names=self.invalid_names, top=False)

            if isinstance(subfragment, ir.Instance):
                submodule, ports = self._process_submodule_instance(submodule)
            else:
                submodule.prepare()
                ports = None
                for port in submodule.ports:
                    local_signal = self._signals.get(port.signal, None)
                    if local_signal is None:
                        local_signal = self._signals[port.signal] = Signal(port.signal)

                    if port.direction in ['o', 'io']:
                        local_signal.disable_reset_statement()

            # TODO: Maybe rename submodule ports as <submodule.name>__<port_name>

            if submodule is not None:
                self.submodules.append((submodule, ports))

class MemoryModule(Module):
    def __init__(self, name, fragment, hdl=None, invalid_names=None, top=True):
        super().__init__(name, fragment, hdl=hdl, invalid_names=invalid_names, top=top)
        self.invalid_names.pop()

        self._mem = self._read_proxy = None

        self._has_read = fragment.parameters.get('RD_CLK_ENABLE', ast.Const(0, 1)).value
        if self._has_read:
            rclk = fragment.named_ports.get('RD_CLK', None)
            if rclk is None:
                raise RuntimeError(f"Missing read clock for memory {self.name}")

            if isinstance(rclk[0], ast.Const) and rclk[0].value == 0:
                self._rdom = None
            else:
                self._rdom = self._find_domain_from_clock(self._process_rhs(rclk[0]))

            rindex = fragment.named_ports.get('RD_ADDR', None)
            if rindex is None:
                raise RuntimeError(f"Failed to find read address for memory {self.name}")
            self._rindex = self._process_rhs(rindex[0])

            r_en = fragment.named_ports.get('RD_EN', None)
            if r_en is None:
                raise RuntimeError(f"Missing write enable for memory {self.name}")
            self._r_en = self._process_rhs(r_en[0])
        else:
            self._rdom = self._rindex = None

        self._has_write = fragment.parameters.get('WR_CLK_ENABLE', ast.Const(0, 1)).value
        if self._has_write:
            wclk = fragment.named_ports.get('WR_CLK', None)
            if wclk is None:
                raise RuntimeError(f"Missing read clock for memory {self.name}")
            self._wdom = self._find_domain_from_clock(self._process_rhs(wclk[0]))
            windex = fragment.named_ports.get('WR_ADDR', None)
            if windex is None:
                raise RuntimeError(f"Failed to find write address for memory {self.name}")
            self._windex = self._process_rhs(windex[0])

            w_en = fragment.named_ports.get('WR_EN', None)
            if w_en is None:
                raise RuntimeError(f"Missing write enable for memory {self.name}")
            self._w_en = self._process_rhs(w_en[0])

        else:
            self._wdom = self._windex = None

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
        raise RuntimeError(f"Failed to find domain for clock {clock}")

    def _prepare_signals(self):
        super()._prepare_signals()

        # TODO: Handle multi-port and granularity!!!

        self._mem = self._signals[self._new_signal(
            width   = self._width,
            prefix  = self.name,
            mapping = Memory,
            domain  = self._wdom.name,
            w_index = self._windex,
            r_index = self._rindex,
            init    = self._init,
        )]

        if self._has_read:
            self._read_proxy = self._new_signal(width = self._width, prefix = f'{self.name}_rdata', domain = self._rdom.name)
            self._add_new_statement(self._read_proxy, Assign(self._mem))

        self._update_statements(self.fragment.statements)

    def _update_statements(self, statements, rst_cond = False, elems = None):
        replace_statements = []
        if elems is None:
            elems = ast.SignalSet()

        for i, st in enumerate(statements):
            if isinstance(st, ast.Assign):
                arr = None
                if isinstance(st.lhs, ast.ArrayProxy):
                    
                    if not self._has_write:
                        raise RuntimeError(f"Read-only memory {self.name} at left hand side is invalid!")

                    arr = st.lhs
                    st.lhs = self._mem.signal
                    if rst_cond:
                        replace_statements.append((i, None))
                elif isinstance(st.rhs, ast.ArrayProxy):
                    if not self._has_read:
                        raise RuntimeError(f"Write-only memory {self.name} at right hand side is invalid!")

                    arr = st.rhs
                    st.rhs = self._read_proxy

                if arr is not None:
                    for elem in arr.elems:
                        elems.add(elem)
                        self._signals.pop(elem, None)

                for s in [st.lhs, st.rhs]:
                    if isinstance(s, ast.Signal) and s in elems:
                        replace_statements.append((i, None))
                        break

            elif isinstance(st, ast.Switch):
                if st.test is self._r_en:
                    assert [('1',)] == list(st.cases), f"Invalid memory statement for module {self.name}"
                    new_list = st.cases[('1',)]
                    self._update_statements(new_list, elems=elems)
                    replace_statements.append((i, ast._StatementList(new_list)))
                else:
                    rst_cond = (self._rdom is not None and st.test is self._rdom.rst) or (self._wdom is not None and st.test is self._wdom.rst)
                    for stmts in st.cases.values():
                        self._update_statements(stmts, rst_cond=rst_cond, elems=elems)

        for i, new in reversed(replace_statements):
            if new is None:
                statements.pop(i)
            else:
                statements[i] = new

