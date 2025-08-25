from amaranth.hdl import ast, ir
import celosia.backend.signal as celosia_signal
import celosia.backend.statement as celosia_statement
from typing import Union

class Module:

    allow_remapping = True  # Not strictly necessary, but cocotb needs it to force-assign some signals

    def __init__(self, name: str, fragment: ir.Fragment, top: bool = True, type: str = None):
        self.top = top

        self.name = name
        self.type = name if type is None else type
        self.fragment = fragment
        self.submodules: list[Module] = []
        self.signals: dict[ast.Signal, celosia_signal.Signal] = ast.SignalDict()
        self.ports: list[celosia_signal.Port] = []
        self._remapped = ast.SignalDict()

    @property
    def empty(self) -> bool:
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
    def domains(self) -> dict:
        return self.fragment.domains

    @property
    def parameters(self) -> dict:
        return {}

    def _reset(self) -> None:
        self.signals.clear()
        self.submodules.clear()
        self.ports.clear()
        self._remapped.clear()

    def _get_signal(self, signal: ast.Signal):
        s = self.signals.get(signal, None)
        if s is None:
            raise RuntimeError(f"Missing signal {signal.name} from module {self.name}")
        return s

    def _new_signal(self, shape: Union[int, ast.Shape] = 1, prefix: str = None, mapping: type = celosia_signal.Signal, **kwargs) -> ast.Value:
        name = prefix or 'tmp'
        new = ast.Signal(shape, name=name)
        self.signals[new] = mapping(new, **kwargs)

        return new

    def _zero_size_signal(self) -> ast.Value:
        # TODO: Check
        return ast.Const(0, 0)
        return self._new_signal(0, prefix = 'empty')

    def _add_new_statement(self, left: ast.Value, statement: celosia_statement.Statement) -> None:
        ########################
        if left not in self.signals:
            self.signals[left] = celosia_signal.Signal(left)
        ########################
        remap = self._remapped.get(left, None)
        if remap is not None:
            left = remap
        self._get_signal(left).add_statement(statement)

    def _add_new_assign(self, left: ast.Signal, right: ast.Value, start_idx: int = None, stop_idx: int = None) -> None:
        self._add_new_statement(left, celosia_statement.Assign(right, start_idx, stop_idx))

    def prepare(self) -> "Module":
        self._prepare_signals()
        self._prepare_statements()
        self._prepare_submodules()
        return self

    def _prepare_signals(self) -> None:

        # TODO: Possibly create intermediate signals so that ports are always wire
        for port, direction in self.fragment.ports.items():
            self.signals[port] = celosia_signal.Port(port, direction=direction)

            # FIX: Zero-width ports are skipped early
            if len(port):
                self.ports.append(self.signals[port])

        for domain, signal in self.fragment.iter_drivers():
            if domain is not None:
                domain = self.domains[domain]

            entry = self.signals.get(signal, None)
            if entry is None:
                entry = self.signals[signal] = celosia_signal.Signal(signal)

            if self.allow_remapping and domain is not None:
                remap = self._remapped[signal] = self._new_signal(
                    signal.shape(),
                    prefix = f'{signal.name}_next',
                    mapping = celosia_signal.RemappedSignal,
                    sync_signal = signal,
                )

                # For some reason, cocotb needs rst duplication in comb and sync parts
                if domain.rst is None or not domain.async_reset:
                    entry.add_statement(celosia_statement.Assign(remap))
                else:
                    entry.add_statement(celosia_statement.Switch(domain.rst, {
                        '0': [celosia_statement.Assign(remap)],
                        '1': [celosia_statement.Assign(ast.Const(signal.reset, len(signal)))]
                    }))

            entry.domain = domain

    @staticmethod
    def _slice_check_const(rhs: ast.Value, start: int, stop: int) -> ast.Value:
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

    def _process_lhs(self, lhs: ast.Value, rhs: ast.Value, start_idx: int = None, stop_idx: int = None) -> list[tuple]:
        res: list[tuple] = []

        # TODO: Review start_idx/stop_idx, maybe it's better to use intermediate signals!

        if start_idx is None or start_idx < 0:
            start_idx = 0
        if stop_idx is None or stop_idx > len(lhs):
            stop_idx = len(lhs)

        if isinstance(lhs, (ast.Const, ast.Operator)):
            raise RuntimeError("Invalid LHS:", lhs)
        elif isinstance(lhs, ast.Signal):
            res.append((lhs, celosia_statement.Assign(rhs, start_idx, stop_idx)))
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

    @classmethod
    def _open_concat(cls, concat: ast.Cat):
        new_parts = []

        for part in concat.parts:
            if not len(part):
                continue

            if isinstance(part, ast.Cat):
                cls._open_concat(part)
                new_parts.extend(part.parts)
            else:
                new_parts.append(part)

        concat.parts = new_parts

    def _process_rhs(self, rhs: ast.Value, **kwargs) -> ast.Value:
        # TODO: Possibly check if return value differs input value to determine whether a new signal is needed
        # so we can reduce code size

        # TODO: Possibly receive a "top" parameter, so that the first layer doesn't need to create a new signal
        # For example: assign x = Cat(Slice, Part) ---> x = Cat(new_slice, new_part) instead of x = new_signal_for_cat

        _division_fix = kwargs.get('_division_fix', False)
        io = kwargs.get('io', False)

        if isinstance(rhs, ast.Const):
            pass
        elif isinstance(rhs, ast.Signal):
            # Fix: Can happen with submodule ports
            if rhs not in self.signals:
                self.signals[rhs] = celosia_signal.Signal(rhs)

        elif isinstance(rhs, ast.Cat):
            self._open_concat(rhs)
            parts = [part for part in rhs.parts if len(part)]
            if len(parts) == 0:
                rhs = self._zero_size_signal()
            elif len(parts) == 1:
                rhs = self._process_rhs(parts[0], **kwargs)
            else:
                rhs.parts = [self._process_rhs(part, **kwargs) for part in parts]

                if not io:
                    new_rhs = self._new_signal(rhs.shape(), prefix='concat')
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
                elif not io:
                    new_rhs = self._new_signal(rhs.shape(), prefix='slice')
                    self._add_new_assign(new_rhs, rhs)
                    rhs = new_rhs

        elif isinstance(rhs, ast.Operator):
            for i, operand in enumerate(rhs.operands):
                rhs.operands[i] = self._process_rhs(operand, **kwargs)

            if io:
                raise RuntimeError(f"Invalid assignment for IO: {rhs}")

            if not _division_fix and rhs.operator == '//' and len(rhs.operands) == 2:
                dividend, divisor, signed = self._signed_division_fix(rhs)
                kwargs['_division_fix'] = True

                if signed:
                    signed = lambda x: ast.signed(x)
                else:
                    signed = lambda x: x
                new_rhs = self._process_rhs(ast.Mux(divisor == ast.Const(0, signed(len(divisor))), ast.Const(0, signed(len(rhs))), dividend//divisor), **kwargs)

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
                if io:
                    raise RuntimeError("Can't generate IO with Array")

                for i, elem in enumerate(rhs.elems):
                    rhs.elems[i] = self._process_rhs(elem, **kwargs)

                rhs.index = self._process_rhs(rhs.index, **kwargs)

                new_rhs = self._new_signal(rhs.shape(), prefix='array')

                index = self._process_rhs(rhs.index, **kwargs)
                cases = {
                    i: [celosia_statement.Assign(elem)] for i, elem in enumerate(rhs.elems)
                }

                # if 2**len(index) > len(rhs.elems):
                #     cases[None] = 0 # Default

                self._add_new_statement(new_rhs, celosia_statement.Switch(index, cases))
                rhs = new_rhs
        else:
            raise ValueError("Unknown RHS object detected: {}".format(rhs.__class__.__name__))

        return rhs

    def _signed_division_fix(self, rhs: ast.Operator) -> tuple[ast.Value, ast.Value]:
        dividend, divisor = rhs.operands

        signed = False

        if any(operand.shape().signed for operand in rhs.operands):
            signed = True

            max_size = max(len(op) for op in rhs.operands) + 2

            if not dividend.shape().signed:
                dividend = dividend.as_signed()

            if not divisor.shape().signed:
                divisor = divisor.as_signed()

            dividend = self._fix_rhs_size(dividend, max_size)
            divisor = self._fix_rhs_size(divisor, max_size)

            # Yosys fix for signed division
            dividend = self._fix_rhs_size(ast.Mux(
                (dividend[-1] == divisor[-1]) | (dividend == ast.Const(0, len(dividend))),
                dividend,
                dividend - ast.Mux(divisor[-1], divisor + ast.Const(1, len(divisor)), divisor - ast.Const(1, len(divisor)))
            ), size = len(dividend))

        return dividend, divisor, signed

    def _fix_rhs_size(self, rhs: ast.Value, size: int = None, *, _force_sign: bool = None, _allow_upsize: bool = False):
        if size is None:
            size = len(rhs)

        # TODO: When the RHS is for example +, we need to know that the result will have +1 bit
        # so we don't need to resize with max(size, max(len(operands)))... We should include that +1 to avoid
        # upsizing signals unnecessarily

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
                self._add_new_assign(new_rhs, self._fix_rhs_size(rhs, size), stop_idx=size)
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
                    rhs.elems[i] = self._fix_rhs_size(elem, size)

            if isinstance(rhs, (ast.Signal, ast.Cat, ast.Slice, ast.Part, ast.ArrayProxy)):
                if len(rhs) < size or (len(rhs) > size and not _allow_upsize):
                    new_rhs = self._new_signal(ast.Shape(size, signed=rhs.shape().signed), prefix = 'resized')
                    self._add_new_assign(new_rhs, self._fix_rhs_size(rhs), stop_idx=size)
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
                        self._add_new_assign(new_rhs, rhs, stop_idx=size)
                        rhs = new_rhs
                    else:
                        signed = any(op.shape().signed for op in operands)
                        for i, operand in enumerate(rhs.operands):
                            rhs.operands[i] = self._fix_rhs_size(operand, _allow_upsize=True, _force_sign=signed)

                        signed = any(op.shape().signed for op in rhs.operands)

                        # TODO: Find a way to avoid using size here to remove unnecessary extra bits
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
                    self._add_new_assign(new_rhs, rhs, stop_idx=size)
                    rhs = new_rhs

            else:
                raise ValueError("Unknown RHS object detected: {}".format(rhs.__class__.__name__))

        return self._process_rhs(rhs)

    def _open_switch(self, test: ast.Value, cases: dict) -> list[tuple[ast.Signal, celosia_statement.Switch]]:
        res = []
        per_signal = ast.SignalDict()

        allsignals = ast.SignalSet()
        for case, statements in cases.items():
            for signal, st in statements:
                allsignals.add(signal)

        for case, statements in cases.items():
            for signal in allsignals:
                per_signal.setdefault(signal, {}).setdefault(case, [])

            for signal, st in statements:
                per_signal[signal][case].append(st)

        test = self._fix_rhs_size(test)

        res = []
        for signal, cases in per_signal.items():
            switch = celosia_statement.Switch(test, cases)

            if not switch.cases:
                continue

            if len(switch.cases) == 1 and None in switch.cases:
                res.extend(((signal, statement) for statement in switch.cases[None]))
            else:
                res.append((signal, switch))

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

    def _process_statement(self, statement: ast.Statement):
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

    def _execute_statements(self, statements: list[ast.Statement]):
        for statement in statements:
            for signal, st in self._process_statement(statement):
                self._add_new_statement(signal, st)

    def _prepare_statements(self):
        self._execute_statements(self.fragment.statements)

    def _submodule_create(self, name: str, fragment: ir.Fragment, cls: type = None, **kwargs) -> "Module":
        if cls is None:
            cls = Module

        return cls(name, fragment, top=False, **kwargs)

    def _process_memory(self, subfragment: ir.Fragment, name: str):
        m = self._submodule_create(name, subfragment, MemoryModule).prepare()

        for signal, mapping in m.signals.items():

            if isinstance(mapping, (celosia_signal.Memory, celosia_signal.MemoryPort)):
                assert signal not in self.signals, "Internal error with Memory signals"
                self.signals[signal] = mapping

            else:
                curr_mapping = self.signals.get(signal, None)
                if curr_mapping is None:
                    self.signals[signal] = curr_mapping = celosia_signal.Signal(mapping.signal, mapping.domain)

                if curr_mapping.domain is None:
                    curr_mapping.domain = mapping.domain    # Allow for domain changes

                for statement in mapping.statements:
                    self.signals[signal].add_statement(statement)

    def _process_submodule_instance(self, subfragment: ir.Fragment, name: str) -> "Module":
        if subfragment.type == "$mem_v2":   # TODO: Check if there's a better way to determine this
            self._process_memory(subfragment, name)
            submodule = None
        else:
            submodule = self._submodule_create(name, subfragment, InstanceModule).prepare()
            for i, (port_name, (port_value, kind)) in enumerate(subfragment.named_ports.items()):
                local_signals: list[celosia_signal.Signal] = []

                try:
                    # We're here for IO, but for I and O ports tied to constants/signals/slice/cat we can reduce some logic
                    port = self._process_rhs(port_value, io=True)
                    if kind in ['o', 'io']:
                        local_signals.extend([self.signals.get(signal, None) for signal in port._rhs_signals()])

                except RuntimeError as e:
                    if kind == 'io':
                        raise e from None

                    # TODO: Better port naming?
                    port = new_port = self._new_signal(port_value.shape(), prefix=f'port_{port_name}')
                    if kind == 'i':
                        self._execute_statements([new_port.eq(port_value)])
                    elif kind == 'o':
                        self._execute_statements([port_value.eq(new_port)])
                        local_signals.append(self.signals[new_port])
                    else:
                        raise RuntimeError(f"Unknown port type for port {port_name} for submodule {submodule.name} of module {self.name}: {kind}")

                submodule.ports[i].signal = port

                for local_signal in local_signals:
                    if local_signal is not None:
                        local_signal.disable_reset_statement()

        return submodule

    def _prepare_submodules(self):
        for subfragment, subname in self.fragment.subfragments:
            if subname is None:
                subname = 'unnamed'

            if isinstance(subfragment, ir.Instance):
                submodule = self._process_submodule_instance(subfragment, subname)
            else:
                submodule: Module = self._submodule_create(subname, subfragment, type = f'{self.type}_{subname}')
                submodule.prepare()
                for port in submodule.ports:
                    local_signal = self.signals.get(port.signal, None)
                    if local_signal is None:
                        local_signal = self.signals[port.signal] = celosia_signal.Signal(port.signal)

                    if port.direction in ['o', 'io']:
                        local_signal.disable_reset_statement()

                    # FIX: Force alt_name early to avoid rename when signals have multiple hierarchy jumps
                    # and submodules are not seen directly
                    port.set_alt_name(port.name)

            if submodule is not None:
                self.submodules.append(submodule)

class InstanceModule(Module):
    def __init__(self, name: str, fragment: ir.Fragment, top: bool = True):
        super().__init__(name, fragment, top=top, type=fragment.type)

    def _prepare_signals(self) -> None:
        super()._prepare_signals()

        self.ports.clear()
        self.signals.clear()
        for port_name, (_, kind) in self.fragment.named_ports.items():
            port = self.signals[self._new_signal(prefix=port_name, mapping=celosia_signal.Port, direction=kind)]
            port.set_alt_name(port_name)
            self.ports.append(port)

    @property
    def parameters(self):
        return self.fragment.parameters

class MemoryModule(Module):

    allow_remapping = False

    class Port:
        def __init__(self, data: ast.Signal, index: ast.Signal, enable: ast.Signal, domain: ir.ClockDomain):
            self.data = data
            self.index = index
            self.enable = enable
            self.domain = domain

            self.proxy: ast.Signal = None

        @staticmethod
        def _get_signal(name: str, n: int, ports: dict[str, ast.Value]) -> list[ast.Value]:
            signal = ports.get(name, None)
            if name is None:
                raise RuntimeError(f"Failed to find signal {name} for memory port")
            signal = signal[0]
            assert isinstance(signal, ast.Cat), f"Port {name} for memory port must be a concatenation"
            assert len(signal.parts) == n, f"Port {name} for memory port has incorrect width ({len(signal.parts)} != {n})"
            return signal.parts

        @classmethod
        def from_fragment(cls, fragment: ir.Instance, prefix: str, domain_resolver) -> list["MemoryModule.Port"]:
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
        def from_fragment(cls, fragment: ir.Instance, domain_resolver) -> list["MemoryModule.ReadPort"]:
            return super().from_fragment(fragment, 'RD', domain_resolver)

    class WritePort(Port):
        @classmethod
        def from_fragment(cls, fragment: ir.Instance, domain_resolver) -> list["MemoryModule.WritePort"]:
            return super().from_fragment(fragment, 'WR', domain_resolver)

    def __init__(self, name: str, fragment: ir.Instance, top: bool = True):
        super().__init__(name, fragment, top=top, type=fragment.type)

        self._mem: celosia_signal.Memory = None
        self._arr = ast.SignalSet()

        self._r_ports: list[MemoryModule.ReadPort] = self.ReadPort.from_fragment(
            fragment, domain_resolver=self._find_domain_from_clock
        )
        self._w_ports: list[MemoryModule.WritePort] = self.WritePort.from_fragment(
            fragment, domain_resolver=self._find_domain_from_clock
        )

        self._width: int = fragment.parameters.get('WIDTH', None)
        if self._width is None:
            raise RuntimeError(f"Failed to find width for memory {self.name}")

        self._size: int = fragment.parameters.get('SIZE', None)
        if self._size is None:
            raise RuntimeError(f"Failed to find size for memory {self.name}")

        init: int = fragment.parameters.get('INIT', ast.Const(0, self._width * self._size)).value
        self._init = [
            ast.Const((init >> (self._width * i)) & int('1' * self._width, 2), self._width) for i in range(self._size)
        ]

    def _find_domain_from_clock(self, clock: ast.Signal) -> ir.ClockDomain:
        for domain in self.domains.values():
            if domain.clk is clock:
                return domain
        raise RuntimeError(f"Failed to find domain for clock {clock} of {self.name}")

    def _prepare_signals(self):
        super()._prepare_signals()

        # TODO: Handle granularity!!!

        self._mem = self.signals[self._new_signal(
            shape   = self._width,
            prefix  = self.name,
            mapping = celosia_signal.Memory,
            init    = self._init,
        )]


        for rport in self._r_ports:
            self.signals[rport.data].domain = rport.domain
            rport.proxy = celosia_signal.MemoryPort(self._mem.signal, memory = self._mem, index = rport.index)

            if rport.domain is None:  # TODO: Also here if r_en is never assigned by parent module
                self.signals.pop(rport.enable, None)

        for wport in self._w_ports:
            wport.proxy = self._new_signal(
                shape = self._width,
                prefix = f'{self.name}_internal',    # Doesn't matter, signal won't really exist
                mapping = celosia_signal.MemoryPort,
                domain = wport.domain,
                memory = self._mem,
                index = wport.index,
            )
            self.signals[wport.proxy].signal = self._mem.signal    # Hacky!

        self._update_statements(self.fragment.statements)

    def _reset(self):
        super()._reset()
        self._arr.clear()

    def _set_arr(self, arr: ast.ArrayProxy):
        for signal in arr.elems:
            self.signals.pop(signal, None)

        self._arr.update(arr.elems)

    def _process_rhs(self, rhs: ast.Value, **kwargs) -> ast.Value:
        if isinstance(rhs, ast.ArrayProxy):
            self._set_arr(rhs)

            for rport in self._r_ports:
                if rhs.index is rport.index:
                    return rport.proxy

            raise RuntimeError(f"Port read index not found for memory {self.name}")

        if self._mem.signal in rhs._rhs_signals():
            return rhs

        return super()._process_rhs(rhs, **kwargs)

    def _process_lhs(self, lhs: ast.Value, rhs: ast.Value, start_idx: int = None, stop_idx: int = None) -> list[tuple[ast.Signal, ast.Value]]:
        if isinstance(lhs, ast.ArrayProxy):
            self._set_arr(lhs)

            for wport in self._w_ports:
                if lhs.index is wport.index:
                    return [(wport.proxy, celosia_statement.Assign(rhs))]

            raise RuntimeError(f"Port write index not found for memory {self.name}")

        elif isinstance(lhs, ast.Signal) and self._arr is not None and lhs in self._arr:
            return []

        return super()._process_lhs(lhs, rhs, start_idx, stop_idx)

    def _update_statements(self, statements: list[ast.Statement]):
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