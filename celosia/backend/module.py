from amaranth.hdl import _ast, _nir, _ir, _cd
import celosia.backend.signal as celosia_signal
import celosia.backend.statement as celosia_statement
from typing import Union

# TODO: If we find multiple signals with same statements, maybe we can merge them into one!

class Module:

    allow_remapping = True  # Not strictly necessary, but cocotb needs it to force-assign some signals

    def __init__(self, name: str, emitter: _ir.NetlistEmitter, module_idx = 0, type: str = None):
        self.name = name
        self.type = name if type is None else type

        self.module = emitter.netlist.modules[module_idx]
        self.emitter = emitter

        self.submodules: list[Module] = []
        self.signals: dict[_ast.Signal, celosia_signal.Signal] = _ast.SignalDict()
        self.ports: list[celosia_signal.Port] = []
        self._remapped = _ast.SignalDict()

        self._value_map: dict[_nir.Net, tuple[_ast.Signal, int]] = {}

    @property
    def empty(self) -> bool:

        # TODO: Maybe cached?
        def _check(module: _nir.Module) -> bool:
            is_empty = not self.module.cells
            for submodule in module.submodules:
                is_empty &= _check(submodule)
            return is_empty

        return _check(self.module)

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

    def _get_signal(self, signal: _ast.Signal):
        s = self.signals.get(signal, None)
        if s is None:
            raise RuntimeError(f"Missing signal {signal.name} from module {self.name}")
        return s

    def _new_signal(self, shape: Union[int, _ast.Shape] = 1, prefix: str = None, mapping: type = celosia_signal.Signal, **kwargs) -> _ast.Value:
        name = prefix or 'tmp'
        new = _ast.Signal(shape, name=name)
        self.signals[new] = mapping(new, **kwargs)

        return new

    def _zero_size_signal(self) -> _ast.Value:
        # TODO: Check
        return _ast.Const(0, 0)
        return self._new_signal(0, prefix = 'empty')

    def _add_new_statement(self, left: _ast.Value, statement: celosia_statement.Statement) -> None:
        ########################
        if left not in self.signals:
            self.signals[left] = celosia_signal.Signal(left)
        ########################
        remap = self._remapped.get(left, None)
        if remap is not None:
            left = remap
        self._get_signal(left).add_statement(statement)

    def _add_new_assign(self, left: _ast.Signal, right: _ast.Value, start_idx: int = None, stop_idx: int = None) -> None:
        self._add_new_statement(left, celosia_statement.Assign(right, start_idx, stop_idx))

    def prepare(self) -> "Module":
        self._prepare_signals()
        self._prepare_statements()
        self._prepare_submodules()
        return self

    def _prepare_signals(self) -> None:
        for key, value in self.emitter.late_net_to_signal.items():
            self._value_map[key] = value

        for signal, name in self.module.signal_names.items():
            signal: _ast.Signal

            for i, part in enumerate(self.emitter.netlist.signals[signal]):
                self._value_map[part] = (signal, i)

            port_entry = self.module.ports.get(name, None)

            if port_entry is not None:
                _, flow = port_entry

                if flow == _nir.ModuleNetFlow.Input:
                    dir = 'i'
                elif flow == _nir.ModuleNetFlow.Output:
                    dir = 'o'
                else:
                    raise RuntimeError(f"Invalid flow for port {name} of module {self.name}: {flow}")

                if len(signal):
                    self.signals[signal] = mapping = celosia_signal.Port(signal, direction=dir)
                    self.ports.append(mapping)
                else:
                    continue

            else:
                self.signals[signal] = mapping = celosia_signal.Signal(signal, domain=domain)
    
            domain: _cd.ClockDomain = tuple(self.emitter.drivers.get(signal, ((None, None),)))[0][1]

            if self.allow_remapping and domain is not None:
                remap = self._remapped[signal] = self._new_signal(
                    signal.shape(),
                    prefix = f'{signal.name}_next',
                    mapping = celosia_signal.RemappedSignal,
                    sync_signal = signal,
                )

                # For some reason, cocotb needs rst duplication in comb and sync parts
                if domain.rst is None or not domain.async_reset:
                    mapping.add_statement(celosia_statement.Assign(remap))
                else:
                    mapping.add_statement(celosia_statement.Switch(domain.rst, {
                        '0': [celosia_statement.Assign(remap)],
                        '1': [celosia_statement.Assign(_ast.Const(signal.reset, len(signal)))]
                    }))

            mapping.domain = domain

    @staticmethod
    def _slice_check_const(rhs: _ast.Value, start: int, stop: int) -> _ast.Value:
        if isinstance(rhs, _ast.Const):
            value = rhs.value
            width = min(rhs.width, stop) - start
            # signed = rhs.signed
            if rhs.value < 0:
                value += 2**rhs.width
            value = (value >> start) & int('1' * (stop - start), 2)
            # if signed:
            #     value -= 2**width

            return _ast.Const(value, width)

        else:
            return rhs[start : stop]

    def _process_lhs(self, lhs: _ast.Value, rhs: _ast.Value, start_idx: int = None, stop_idx: int = None) -> list[tuple[_ast.Signal, celosia_statement.Statement]]:
        res: list[tuple] = []

        # TODO: Review start_idx/stop_idx, maybe it's better to use intermediate signals!

        if start_idx is None or start_idx < 0:
            start_idx = 0
        if stop_idx is None or stop_idx > len(lhs):
            stop_idx = len(lhs)

        if isinstance(lhs, _ast.Const):
            raise RuntimeError(f"Invalid LHS, can't be const: {lhs}")
        elif isinstance(lhs, _ast.Signal):
            res.append((lhs, celosia_statement.Assign(rhs, start_idx, stop_idx)))
        elif isinstance(lhs, _ast.Cat):
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
                elif isinstance(rhs, _ast.Slice):
                    # FIX: Avoid duplicating slices!
                    new_rhs = _ast.Slice(rhs.value, rhs.start + roffset, min(rhs.start + roffset + len(part), rhs.stop))
                else:
                    new_rhs = self._slice_check_const(rhs, roffset, roffset + len(part))

                res.extend(self._process_lhs(part, new_rhs, new_start, new_stop))
                loffset += len(part)
                roffset += len(part)

        elif isinstance(lhs, _ast.Slice):
            if lhs.start < lhs.stop:
                start = start_idx+lhs.start
                res.extend(self._process_lhs(lhs.value, rhs, start_idx=start, stop_idx=min(start+stop_idx, lhs.stop)))

        elif isinstance(lhs, _ast.Operator):
            if len(lhs.operands) == 1 and lhs.operator in ('u', 's'):
                res.extend(self._process_lhs(lhs.operands[0], rhs, start_idx=start_idx, stop_idx=stop_idx))
            else:
                raise RuntimeError(f"Invalid LHS, can't be operator: {lhs}")

        elif isinstance(lhs, _ast.Part):
            if isinstance(lhs.offset, _ast.Const):
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

        elif isinstance(lhs, _ast.ArrayProxy):
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
            raise ValueError(f"Unknown RHS object detected: {rhs.__class__.__name__}")

        return res

    @classmethod
    def _open_concat(cls, concat: _ast.Cat):
        new_parts = []

        for part in concat.parts:
            if not len(part):
                continue

            if isinstance(part, _ast.Cat):
                cls._open_concat(part)
                new_parts.extend(part.parts)
            else:
                new_parts.append(part)

        concat.parts = new_parts

    def _process_rhs(self, rhs: _ast.Value, shape: Union[int, _ast.Shape] = None, ignore_sign=False, **kwargs) -> _ast.Value:
        # TODO: Possibly check if return value differs input value to determine whether a new signal is needed
        # so we can reduce code size

        # TODO: Possibly receive a "top" parameter, so that the first layer doesn't need to create a new signal
        # For example: assign x = Cat(Slice, Part) ---> x = Cat(new_slice, new_part) instead of x = new_signal_for_cat
        curr_shape: _ast.Shape = rhs.shape()

        if shape is None:
            shape = curr_shape

        shape: _ast.Shape = _ast.Shape.cast(shape)

        if ignore_sign:
            shape.signed = curr_shape.signed

        io = kwargs.get('io', False)

        if isinstance(rhs, _ast.Const):
            return _ast.Const(rhs.value, shape)

        if curr_shape.width > shape.width:
            return self._process_rhs(_ast.Slice(rhs, 0, shape.width), shape, **kwargs)
        elif curr_shape.width < shape.width:
            new_rhs = self._new_signal(shape, prefix = 'resized')
            self._add_new_assign(new_rhs, self._process_rhs(rhs, **kwargs))
            return new_rhs

        if curr_shape.signed != shape.signed:
            new_rhs = self._new_signal(shape, prefix='resigned')
            self._add_new_assign(new_rhs, self._process_rhs(rhs, **kwargs))
            return new_rhs

        if isinstance(rhs, _ast.Signal):
            # Fix: Can happen with submodule ports
            if rhs not in self.signals:
                self.signals[rhs] = celosia_signal.Signal(rhs)

        elif isinstance(rhs, _ast.Cat):
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

        elif isinstance(rhs, _ast.Slice):
            if rhs.start >= rhs.stop:
                rhs = self._zero_size_signal()
            else:
                rhs.value = self._process_rhs(rhs.value, **kwargs)

                if rhs.start == 0 and rhs.stop >= len(rhs.value):
                    rhs = rhs.value
                elif isinstance(rhs.value, _ast.Const):
                    rhs = self._slice_check_const(rhs.value, rhs.start, rhs.stop)
                elif not io:
                    new_rhs = self._new_signal(rhs.shape(), prefix='slice')
                    self._add_new_assign(new_rhs, rhs)
                    rhs = new_rhs

        elif isinstance(rhs, _ast.Operator):

            if io:
                raise RuntimeError(f"Invalid assignment for IO: {rhs}")

            shapes: list[_ast.Shape] = [op.shape() for op in rhs.operands]

            if len(rhs.operands) == 1:
                if rhs.operator in ['u', 's']:
                    signed = rhs.operator == 's'
                    return self._process_rhs(rhs.operands[0], _ast.Shape(rhs.shape().width, signed), **kwargs)
                else:
                    rhs.operands = [self._process_rhs(op, **kwargs) for op in rhs.operands]

            elif len(rhs.operands) == 2:
                if rhs.operator in ('<<', '>>', '**'):
                    new_shape = None

                    # TODO: This can be uncommented to reduce some logic
                    # op0, op1 = rhs.operands
                    # if isinstance(op1, _ast.Const):
                    #     if rhs.operator == '<<':
                    #         return self._process_rhs(_ast.Cat(_ast.Const(0, op1.value), op0), shape)
                    #     elif rhs.operator == '>>':
                    #         return self._process_rhs(_ast.Slice(op0, op1.value, len(op0)), shape)
                elif shapes[0].signed == shapes[1].signed:
                    new_shape = _ast.Shape(max(
                        shapes[0].width,
                        shapes[1].width,
                    ), shapes[0].signed)
                else:
                    new_shape = _ast.signed(max(
                        shapes[0].width + shapes[1].signed,
                        shapes[1].width + shapes[0].signed,
                    ))

                rhs.operands = [self._process_rhs(op, new_shape, **kwargs) for op in rhs.operands]

            elif len(rhs.operands) == 3:
                assert rhs.operator == 'm'

                sel = rhs.operands[0]

                if len(sel) != 1:
                    sel = sel.bool()

                new_shape = rhs.shape()
                rhs.operands = [
                    self._process_rhs(sel, **kwargs),
                    *[self._process_rhs(op, new_shape, **kwargs) for op in rhs.operands[1:]],
                ]

            else:
                raise ValueError(f"Unknown number of operands: {len(rhs.operands)}")

            if rhs.operator in ('//', '%') and len(rhs.operands) == 2:
                rhs = self._division_fix(rhs, **kwargs)

            new_rhs = self._new_signal(rhs.shape(), prefix='operand')
            self._add_new_assign(new_rhs, rhs)
            rhs = new_rhs

            # TODO: Improve, will fall here almost always
            if len(rhs) > shape.width:
                sliced = self._new_signal(shape, prefix='sliced_op')
                self._add_new_assign(sliced, rhs[:shape.width])
                rhs = sliced

            elif len(rhs) < shape.width:
                raise RuntimeError("This should never happen!")

        elif isinstance(rhs, _ast.Part):
            if isinstance(rhs.offset, _ast.Const):
                raise RuntimeError("Part with const offset is Slice!")
            else:
                shift = rhs.offset

                if rhs.stride > 1:
                    shift = shift * rhs.stride

                rhs = self._process_rhs((rhs.value >> shift)[:rhs.width], **kwargs)   # TODO: This is cleaner, but may cause issues!

        elif isinstance(rhs, _ast.ArrayProxy):
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
            raise ValueError(f"Unknown RHS object detected: {rhs.__class__.__name__}")

        return rhs

    def _division_fix(self, rhs: _ast.Operator, **kwargs) -> _ast.Value:
        dividend, divisor = rhs.operands

        if rhs.shape().signed:
            max_size = max(len(op) for op in rhs.operands) + 2

            dividend = self._process_rhs(dividend, _ast.signed(max_size))
            divisor = self._process_rhs(divisor, _ast.signed(max_size))

            # Yosys fix for signed division
            dividend = self._process_rhs(_ast.Mux(
                (dividend[-1] == divisor[-1]) | (dividend == _ast.Const(0, len(dividend))),
                dividend,
                dividend - _ast.Mux(divisor[-1], divisor + _ast.Const(1, len(divisor)), divisor - _ast.Const(1, len(divisor)))
            ), _ast.signed(len(dividend)))

        if rhs.operator == '//':
            real_div = dividend//divisor
        elif rhs.operator == '%':
            real_div = dividend%divisor
        else:
            raise ValueError(f"Invalid operator for division fix: {rhs.operator}")

        rhs_div = self._new_signal(real_div.shape(), prefix='division')
        self._add_new_assign(rhs_div, real_div)

        return self._process_rhs(_ast.Mux(divisor == 0, 0, rhs_div), **kwargs)

    def _open_switch(self, test: _ast.Value, cases: dict) -> list[tuple[_ast.Signal, celosia_statement.Statement]]:
        res = []
        per_signal = _ast.SignalDict()

        allsignals = _ast.SignalSet()
        for case, statements in cases.items():
            for signal, st in statements:
                allsignals.add(signal)

        for case, statements in cases.items():
            for signal in allsignals:
                per_signal.setdefault(signal, {}).setdefault(case, [])

            for signal, st in statements:
                per_signal[signal][case].append(st)

        test = self._process_rhs(test)

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

    def _process_assign(self, assign: _ast.Assign) -> list[tuple[_ast.Signal, celosia_statement.Statement]]:
        return self._process_lhs(assign.lhs, self._process_rhs(assign.rhs, assign.lhs.shape(), ignore_sign=True))

    def _process_switch(self, switch: _ast.Switch) -> list[tuple[_ast.Signal, celosia_statement.Statement]]:
        cases = {}
        for case, statements in switch.cases.items():
            for statement in statements:
                cases.setdefault(case, []).extend(self._process_statement(statement))

        return self._open_switch(switch.test, cases)

    def _process_statement(self, statement: _ast.Statement) -> list[tuple[_ast.Signal, celosia_statement.Statement]]:
        res = []
        if isinstance(statement, _ast.Assign):
            res.extend(self._process_assign(statement))

        elif isinstance(statement, _ast.Switch):
            res.extend(self._process_switch(statement))

        elif isinstance(statement, _ast._StatementList):
            for st in statement:
                res.extend(self._process_statement(st))
        else:
            raise ValueError(f"Unknown statement: {statement.__class__.__name__}")

        return res

    def _execute_statements(self, statements: list[_ast.Statement]):
        for statement in statements:
            for signal, st in self._process_statement(statement):
                self._add_new_statement(signal, st)

    def _prepare_statements(self):
        self._execute_statements(self.fragment.statements)

    def _submodule_create(self, name: str, fragment: _ir.Fragment, cls: type = None, **kwargs) -> "Module":
        if cls is None:
            cls = Module

        return cls(name, fragment, **kwargs)

    def _process_memory(self, subfragment: _ir.Fragment, name: str):
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

    def _process_submodule_instance(self, subfragment: _ir.Instance, name: str) -> "Module":
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

            if isinstance(subfragment, _ir.Instance):
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
    def __init__(self, name: str, fragment: _ir.Instance):
        super().__init__(name, fragment, type=fragment.type)
        self.attrs = fragment.attrs

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
        def __init__(self, data: _ast.Signal, index: _ast.Signal, enable: _ast.Signal, domain: _cd.ClockDomain):
            self.data = data
            self.index = index
            self.enable = enable
            self.domain = domain

            self.proxy: _ast.Signal = None

        @staticmethod
        def _get_signal(name: str, n: int, ports: dict[str, _ast.Value]) -> list[_ast.Value]:
            signal = ports.get(name, None)
            if name is None:
                raise RuntimeError(f"Failed to find signal {name} for memory port")
            signal = signal[0]
            assert isinstance(signal, _ast.Cat), f"Port {name} for memory port must be a concatenation"
            assert len(signal.parts) == n, f"Port {name} for memory port has incorrect width ({len(signal.parts)} != {n})"
            return signal.parts

        @classmethod
        def from_fragment(cls, fragment: _ir.Instance, prefix: str, domain_resolver) -> list["MemoryModule.Port"]:
            ports = fragment.named_ports
            parameters = fragment.parameters

            n = parameters.get(f'{prefix}_PORTS', 0)

            datas = cls._get_signal(f'{prefix}_DATA', n, ports)
            indexes = cls._get_signal(f'{prefix}_ADDR', n, ports)
            enables = cls._get_signal(f'{prefix}_EN', n, ports)

            clk = cls._get_signal(f'{prefix}_CLK', n, ports)
            domains = []

            for i in range(n):
                if isinstance(clk[i], _ast.Const) and clk[i].value == 0:
                    domain = None
                else:
                    domain = domain_resolver(clk[i])

                domains.append(domain)

            return [
                cls(*entry) for entry in zip(datas, indexes, enables, domains)
            ]

    class ReadPort(Port):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._statements = []

        @property
        def statements(self):
            if self.domain is None:
                raise RuntimeError("Combinational read ports don't have statements")
            return self._statements

        @classmethod
        def from_fragment(cls, fragment: _ir.Instance, domain_resolver) -> list["MemoryModule.ReadPort"]:
            return super().from_fragment(fragment, 'RD', domain_resolver)

    class WritePort(Port):
        @classmethod
        def from_fragment(cls, fragment: _ir.Instance, domain_resolver) -> list["MemoryModule.WritePort"]:
            return super().from_fragment(fragment, 'WR', domain_resolver)

    def __init__(self, name: str, fragment: _ir.Instance):
        super().__init__(name, fragment, type=fragment.type)

        self._mem: celosia_signal.Memory = None
        self._arr = _ast.SignalSet()

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

        init: int = fragment.parameters.get('INIT', _ast.Const(0, self._width * self._size)).value
        self._init = [
            _ast.Const((init >> (self._width * i)) & int('1' * self._width, 2), self._width) for i in range(self._size)
        ]

    def _find_domain_from_clock(self, clock: _ast.Signal) -> _cd.ClockDomain:
        for domain in self.domains.values():
            if domain.clk is clock:
                return domain
        raise RuntimeError(f"Failed to find domain for clock {clock} of {self.name}")

    def _prepare_signals(self):
        super()._prepare_signals()

        # TODO: Fix for granularity, memory should be read only once and in the correct domain

        self._mem = self.signals[self._new_signal(
            shape   = self._width,
            prefix  = self.name,
            mapping = celosia_signal.Memory,
            init    = self._init,
            attrs   = self.fragment.attrs,
        )]


        for rport in self._r_ports:
            self.signals[rport.data].domain = None
            read_port = celosia_signal.MemoryPort(self._mem.signal, memory = self._mem, index = rport.index)

            if rport.domain is None:  # TODO: Also here if r_en is never assigned by parent module
                rport.proxy = read_port

                enables = []
                if isinstance(rport.enable, _ast.Cat):
                    enables.extend(rport.enable.parts)
                elif isinstance(rport.enable, _ast.Signal):
                    enables.append(rport.enable)
                elif isinstance(rport.enable, _ast.Const):
                    pass
                else:
                    raise RuntimeError(f"Unknown read enable for memory {self.name}: {rport.enable}")

                for enable in enables:
                    self.signals.pop(enable, None)

            else:
                rport.proxy = self._new_signal(shape = self._width, prefix = '_0_', domain = rport.domain)
                rport.statements.append(celosia_statement.Assign(read_port))

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

    def _prepare_statements(self):
        ret = super()._prepare_statements()

        for rport in self._r_ports:
            if rport.domain is not None:
                self._add_new_statement(rport.proxy, celosia_statement.Switch(rport.enable, {
                    '1': rport.statements,
                }))

        return ret

    def _reset(self):
        super()._reset()
        self._arr.clear()

    def _set_arr(self, arr: _ast.ArrayProxy) -> tuple[int, ...]:
        elems = []
        slices = None
        is_slice = None

        for signal in arr.elems:
            if isinstance(signal, _ast.Slice):
                if is_slice is None:
                    is_slice = True

                if not is_slice:
                    raise RuntimeError(f"Can't mix slice/signal in same array for memory {self.name}")

                new_slice = (signal.start, signal.stop)
                if slices is None:
                    slices = new_slice

                if slices != new_slice:
                    raise RuntimeError(f"Unexpected asignment for memory {self.name}, all array slices should be equal")

                if not isinstance(signal.value, _ast.Signal):
                    raise RuntimeError(f"Invalid array element for memory {self.name}: {signal}")

                signal = signal.value

            elif isinstance(signal, _ast.Signal):
                if is_slice is None:
                    is_slice = False

                if is_slice:
                    raise RuntimeError(f"Can't mix slice/signal in same array for memory {self.name}")

            else:
                raise RuntimeError(f"Invalid array element for memory {self.name}: {signal}")

            self.signals.pop(signal, None)
            elems.append(signal)

        self._arr.update(elems)
        return slices if slices is not None else (None, None)

    def _process_rhs(self, rhs: _ast.Value, shape: Union[int, _ast.Shape] = None, **kwargs) -> _ast.Value:
        if isinstance(rhs, _ast.ArrayProxy):
            self._set_arr(rhs)

            for rport in self._r_ports:
                if rhs.index is rport.index:
                    return rport.proxy

            raise RuntimeError(f"Port read index not found for memory {self.name}")

        elif isinstance(rhs, _ast.Operator) and rhs.operator == 'm' and len(rhs.operands) == 3:
            # FIX: Replace 'mux' with 'if' for transparent=True so we can read memory only once

            if not isinstance(rhs.operands[2], _ast.ArrayProxy):
                raise RuntimeError(f"Unexpected mux assignment in memory {self.name}")

            test = self._process_rhs(rhs.operands[0])
            proxy = self._process_rhs(rhs.operands[2])
            slices = self._set_arr(rhs.operands[2])

            new_statement = celosia_statement.Switch(test, {
                '1': [celosia_statement.Assign(self._process_rhs(rhs.operands[1]), *slices)],
            })
            for rport in self._r_ports:
                if rport.proxy is proxy:
                    rport.statements.append(new_statement)

            return proxy[slice(*slices)]

        if self._mem.signal in rhs._rhs_signals():
            return rhs

        return super()._process_rhs(rhs, shape, **kwargs)

    def _process_lhs(self, lhs: _ast.Value, rhs: _ast.Value, start_idx: int = None, stop_idx: int = None) -> list[tuple[_ast.Signal, _ast.Value]]:
        if isinstance(lhs, _ast.ArrayProxy):
            slices = self._set_arr(lhs)

            for wport in self._w_ports:
                if lhs.index is wport.index:
                    return [(wport.proxy, celosia_statement.Assign(rhs, *slices))]

            raise RuntimeError(f"Port write index not found for memory {self.name}")

        elif isinstance(lhs, _ast.Signal) and self._arr is not None and lhs in self._arr:
            return []

        return super()._process_lhs(lhs, rhs, start_idx, stop_idx)

    def _update_statements(self, statements: list[_ast.Statement]):
        replace_statements = {}

        for i, st in enumerate(statements):
            if not isinstance(st, _ast.Switch):
                continue

            if any(
                st.test is rport.enable for rport in self._r_ports
            ):
                assert [('1',)] == list(st.cases), f"Invalid memory statement for module {self.name}"
                replace_statements[i] = _ast._StatementList(st.cases[('1',)])

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