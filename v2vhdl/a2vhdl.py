from amaranth.hdl import ast, ir
from textwrap import dedent, indent

class HDL:
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

    @staticmethod
    def sanitize(name):
        name = name.strip().replace('\\', '').replace('$', '_esc_').replace('.', '_').replace(':', '_')
        while '__' in name:
            name = name.replace('__', '_')
        if name and name[0] == '_':
            name = 'esc' + name
        if name and name[-1] == '_':
            name = name + 'esc'

        for p in HDL.protected:
            if name == p:
                name = 'esc_' + name

        return name

class Signal:
    def __init__(self, signal, domain = None):
        self.signal = signal
        self.domain = domain
        self.statements = []

    @staticmethod
    def sanitize(name):
        name = HDL.sanitize(name)
        return name

    def add_statement(self, statement):
        if not isinstance(statement, list):
            statement = [statement]
        self.statements.extend(statement)

class Port(Signal):
    def __init__(self, signal, direction, domain=None):
        super().__init__(signal, domain)
        self.direction = direction

# class Memory(Signal):
#     def __init__(self, signal, index, domain=None):
#         super().__init__(signal, domain)
#         self.

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
        self.cases = cases

    @staticmethod
    def convert_case(case):
        if isinstance(case, tuple):
            if len(case) == 1:
                case = Switch.convert_case(case[0])
            elif len(case) == 0:
                case = None
            else:
                raise NotImplementedError("Case {} not supported".format(case))
        elif isinstance(case, str):
            if '-' in case or '?' in case:
                case = case.replace('-', '?')
        elif isinstance(case, int):
            raise ValueError("Integer case needs length!")

        return case

class Module:
    def __init__(self, name, fragment, case_sensitive=False):
        self.name = name
        self.fragment = fragment
        self.case_sensitive = case_sensitive

        self.submodules = {}
        self._signals = ast.SignalDict()

    @staticmethod
    def sanitize_module(name):
        name = HDL.sanitize(name)
        return name

    def _reset(self):
        self._signals.clear()
        self.submodules.clear()

    def _sanitize_signal(self, signal):
        def change_case(name):
            return name if self.case_sensitive else name.lower()

        names = [change_case(signal.name) for signal in self._signals]

        idx = 0
        name = Signal.sanitize(signal.name)
        while change_case(name) in names:
            name = f'{name}{idx}'
            idx += 1

        signal.name = name

    def _get_signal(self, signal):
        s = self._signals.get(signal, None)
        if s is None:
            raise RuntimeError(f"Missing signal {signal.name} from module {self.name}")
        return s

    def _new_signal(self, width=1, prefix=None):
        name = 'tmp'
        if prefix:
            name = f'{prefix}_{name}'
        new = ast.Signal(width, name=name)
        self._sanitize_signal(new)
        self._signals[new] = Signal(new)

        return new

    def _zero_size_signal(self):
        self._new_signal(0, prefix = '$empty')

    def _add_new_statement(self, left, statement):
        self._get_signal(left).add_statement(statement)

    def _add_new_assign(self, left, right, start_idx=None, stop_idx=None):
        self._add_new_statement(left, Assign(right, start_idx, stop_idx))

    def prepare(self, ports=None, platform=None):
        self._reset()
        self.fragment = ir.Fragment.get(self.fragment, platform).prepare(ports)

        self._prepare_signals()
        self._prepare_statements()
        self._prepare_submodules()

    def _prepare_signals(self):
        for port, direction in self.fragment.ports.items():
            self._sanitize_signal(port)
            self._signals[port] = Port(port, direction=direction)

        for domain, signal in self.fragment.iter_drivers():
            entry = self._signals.get(signal, None)
            if entry is None:
                self._sanitize_signal(signal)
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

                    res.extend(self._process_lhs(part, rhs[offset : offset + len(part)], new_start, new_stop))
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
            pass
        elif isinstance(rhs, ast.Cat):
            if len(rhs.parts) == 0:
                rhs = self._zero_size_signal()
            elif len(rhs.parts) == 1:
                rhs = self._process_rhs(rhs.parts[0])
            else:
                new_rhs = self._new_signal(len(rhs), prefix='$concat')
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
                    new_rhs = self._new_signal(len(rhs), prefix='$slice')
                    self._add_new_assign(new_rhs, rhs)
                    rhs = new_rhs

        elif isinstance(rhs, ast.Operator):
            new_rhs = self._new_signal(len(rhs), prefix='$operand')
            for i, operand in enumerate(rhs.operands):
                rhs.operands[i] = self._process_rhs(operand)

            self._add_new_assign(new_rhs, rhs)
            rhs = new_rhs

        elif isinstance(rhs, ast.Part):
            if isinstance(rhs.offset, ast.Const):
                raise RuntimeError("Part with const offset is Slice!")
            else:
                new_rhs = self._new_signal(rhs.stride, prefix='$part')
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

                new_rhs = self._new_signal(max(len(elem) for elem in rhs.elems), prefix='$array')

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

    def _prepare_submodules(self):
        for subfragment, subname in self.fragment.subfragments:
            name = self.sanitize_module(subname)

            submodule = Module(name, subfragment, case_sensitive=self.case_sensitive)

            if isinstance(subfragment, ir.Instance):
                ports = {}
                for port_name, (port_value, kind) in subfragment.named_ports.items():
                    new_port = self._new_signal(len(port_value), prefix=f'port_{port_name}')
                    ports[port_name] = (new_port, kind)
                    if kind == 'i':
                        self._execute_statements([new_port.eq(port_value)])
                    elif kind == 'o':
                        self._execute_statements([port_value.eq(new_port)])
                    elif kind == 'io':
                        pass    # TODO: Check how to handle!
                    else:
                        raise RuntimeError(f"Unknown port type for port {port_name} for submodule {name} of module {self.name}: {kind}")

            else:
                submodule.prepare()
                ports = None

            self.submodules[name] = (submodule, ports)

def v2vhdl(module, name='top', ports=None, blackboxes=None):
    m = Module(name, module, case_sensitive=True)

def main():
    import os
    # with open('test.v', 'r') as f: # with open('test.v', 'r') as f:
    # with open('test2.v', 'r') as f: # with open('test.v', 'r') as f:
    with open(os.path.expanduser('~/Downloads/NOVO/novospace/repos/novohdl/build/top.v'), 'r') as f:
        content = f.read()

    blackboxes = {
        'submodule_type': {
            'p_test0': ('integer', 0),
            'p_test1': ('integer', 2),
            'i_input0': 4,
            'i_input1': 3,
            'o_output0': 8,
        }
    }

    res = v2vhdl(content, blackboxes)

    os.makedirs('testdir', exist_ok=True)

    for key, value in res.items():
        with open(os.path.join('testdir', key), 'w') as f:
            f.write(value)

if __name__ == '__main__':
    main()
