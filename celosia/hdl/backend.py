from amaranth.back import rtlil
from amaranth.back.rtlil import _const
from celosia.hdl.memory import Memory
import celosia.hdl.wire as celosia_wire
from typing import Any, Union
from amaranth.hdl import _ast

# TODO: Tap into rtlil.ModuleEmitter so we can have control over Wire names

class Module(rtlil.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._line: rtlil.Emitter = None

        self._signals: dict[str, rtlil.Wire] = {}

        self._emitted_ports: list[rtlil.Wire] = []
        self._emitted_processes: list[tuple[rtlil.Process, dict]] = []
        self._emitted_submodules: list[rtlil.Cell] = []
        self._emitted_memories: dict[str, Memory] = {}
        self._emitted_operators: list[rtlil.Cell] = []
        self._emitted_flip_flops: list[rtlil.Cell] = []
        self._emitted_divisions: list[rtlil.Cell] = []

        self.name = self.sanitize(self.name)

        self._process_id = 0
        self._rp_count = 0

    @classmethod
    def _const(cls, value: Any):
        return _const(value)

    @classmethod
    def sanitize(cls, name: str) -> str:
        return name

    def _auto_name(self):
        return self.sanitize(super()._auto_name())

    def _name(self, name):
        ret = super()._name(name)
        if ret.startswith('\\'):
            ret = ret[1:]
        return ret

    def wire(self, *args, **kwargs):
        wire = super().wire(*args, **kwargs)
        wire.name = self.sanitize(wire.name)
        self._signals[wire.name] = wire
        if wire.port_kind is not None:
            self._emitted_ports.append(wire)
        return wire

    def process(self, *args, **kwargs):
        process = super().process(*args, **kwargs)
        self._emitted_processes.append((process, {}))
        return process

    def cell(self, *args, **kwargs):
        cell = super().cell(*args, **kwargs)
        cell.name = self.sanitize(cell.name)
        if self._cell_is_submodule(cell):
            self._emitted_submodules.append(cell)
        elif self._cell_is_ff(cell):
            self._emitted_flip_flops.append(cell)
        elif self._cell_is_memory(cell):
            self._get_memory_from_port(cell).set_cell(cell)
        elif self._cell_is_memory_wp(cell):
            self._get_memory_from_port(cell).add_wp(cell)
        elif self._cell_is_memory_rp(cell):
            if self._get_memory_from_port(cell).add_rp(cell, self._rp_count).clk_enable:
                self._rp_count += 1
        elif self._cell_is_division(cell):
            self._emitted_divisions.append(cell)
        # elif self._cell_is_yosys(cell):
        else:
            self._emitted_operators.append(cell)
        return cell

    def memory(self, *args, **kwargs):
        memory = super().memory(*args, **kwargs)
        memory.name = self.sanitize(memory.name)
        self._emitted_memories[memory.name] = self._signals[memory.name] = Memory(memory)
        return memory

    def emit(self, line: rtlil.Emitter):
        self._line = line
        line.port_id = 0

        # Memories need to go first because they may create new signals, processes and connections
        for memory in self._emitted_memories.values():
            self._emit_memory(memory)

        # Divisions may also create new signals, processes and connections
        for division in self._emitted_divisions:
            self._signed_division_fix(division)

        # After this point, no new signals, processes or connections are created
        # We have this callback so subclasses can gather all the information they may need before starting
        self._emit_pre_callback()

        self._emit_module_definition()
        for signal in self._signals.values():
            self._emit_signal(signal)

        for flip_flop in self._emitted_flip_flops:
            self._emit_flip_flop(flip_flop)

        for process, kwargs in self._emitted_processes:
            self._emit_process(process, **kwargs)

        for operator in self._emitted_operators:
            self._emit_operator(operator)

        for submodule in self._emitted_submodules:
            self._emit_submodule(submodule)

        self._emit_connections()
        self._emit_module_end()

        self._line()

    def _emit_pre_callback(self):
        pass

    @classmethod
    def _cell_is_yosys(cls, cell: rtlil.Cell):
        return cell.kind.startswith('$')

    @classmethod
    def _cell_is_submodule(cls, cell: rtlil.Cell):
        return cell.kind.startswith('\\')

    @classmethod
    def _cell_is_ff(cls, cell: rtlil.Cell):
        return cell.kind in ('$dff', '$adff')

    @classmethod
    def _cell_is_memory(cls, cell: rtlil.Cell):
        return cell.kind == '$meminit_v2'

    @classmethod
    def _cell_is_memory_wp(cls, cell: rtlil.Cell):
        return cell.kind == '$memwr_v2'

    @classmethod
    def _cell_is_memory_rp(cls, cell: rtlil.Cell):
        return cell.kind == '$memrd_v2'

    @classmethod
    def _cell_is_division(cls, cell: rtlil.Cell):
        return cell.kind in ('$divfloor', '$modfloor')

    def _get_memory_from_port(self, port: rtlil.Cell) -> Memory:
        memid = port.parameters.get('MEMID', None)
        if memid is None:
            raise RuntimeError(f"MemoryPort {port.name} missing MEMID")

        mem = self._emitted_memories.get(memid, None)
        if mem is None:
            raise RuntimeError(f"Unknown MEMID for MemoryPort {port.name}")

        return mem

    def _new_process(self) -> str:
        ret = f'process_{self._process_id}'
        self._process_id += 1
        return ret

    def _emit_process(self, process: rtlil.Process, comb = True, **kwargs):
        with self._line.indent():
            p_id = self._emit_process_start(**kwargs)

            with self._line.indent():
                self._emit_process_contents(process.contents, comb=comb)

            self._emit_process_end(p_id, comb=comb)

    def _emit_process_start(self, clock: str = None, polarity: bool = True, arst: str = None, arst_polarity = False) -> str:
        return self._new_process()

    def _emit_process_contents(self, contents: list[Union[rtlil.Assignment, rtlil.Switch]], comb=True):
        index = 0
        while index < len(contents) and isinstance(contents[index], rtlil.Assignment):
            self._emit_process_assignment(contents[index], comb=comb)
            index += 1
        while index < len(contents):
            if isinstance(contents[index], rtlil.Assignment):
                pass
                # TODO: What is this?
                # emit(f"switch {{}}")
                # with emit.indent():
                #     emit(f"case")
                #     with emit.indent():
                #         while index < len(contents) and isinstance(contents[index], Assignment):
                #             contents[index].emit(emit)
                #             index += 1
                # emit(f"end")
            else:
                self._emit_switch(contents[index], comb=comb)
                index += 1

    def _emit_process_end(self, p_id: str, comb = True):
        pass

    def _emit_assignment(self, assignment: rtlil.Assignment):
        pass

    def _emit_process_assignment(self, assignment: rtlil.Assignment, comb = True):
        pass

    def _emit_operator_assignment(self, assignment: rtlil.Assignment, comb = True):
        pass

    @classmethod
    def _is_switch_if(cls, switch: rtlil.Switch) -> bool:
        used_idx: set[int] = set()
        for case in switch.cases:
            if not case.patterns:
                if not used_idx:
                    return False    # Weird case with only default?
                break

            if len(case.patterns) > 1:
                return False

            pattern: str = case.patterns[0]
            if pattern.count('1') != 1 or not all(c in ('-', '1') for c in pattern):
                return False

            bit = pattern.index('1')
            if bit in used_idx:
                return False

            used_idx.add(bit)

        return True

    def _emit_switch(self, switch: rtlil.Switch, comb=True):
        # TODO: Filter empty cases
        if self._is_switch_if(switch):
            return self._emit_if(switch, comb=comb)

        self._emit_switch_start(self._represent(switch.sel))

        with self._line.indent():
            for case in switch.cases:
                if case.patterns:
                    pattern = self._case_patterns((self._represent(f"{len(pattern)}'{pattern}") for pattern in case.patterns))
                else:
                    pattern = self._case_default()

                self._emit_case_start(pattern)
                with self._line.indent():
                    self._emit_process_contents(case.contents, comb=comb)
                self._emit_case_end()

        self._emit_switch_end()

    def _emit_if(self, switch: rtlil.Switch, comb=True):
        sel = self._collect_signals(switch.sel)
        first = True

        for case in switch.cases:
            if case.patterns:
                assert len(case.patterns) == 1, "Internal error" # Should never happen
                condition = self._represent(sel[case.patterns[0][::-1].index('1')])
                if first:
                    self._emit_if_start(condition)
                    first = False
                else:
                    self._emit_elseif_start(condition)
            elif not first:
                self._emit_else()

            with self._line.indent():
                self._emit_process_contents(case.contents, comb=comb)

        if not first:
            self._emit_if_end()

    def _emit_switch_start(self, sel: str):
        pass

    def _emit_switch_end(self):
        pass

    def _case_patterns(self, pattern: tuple[str, ...]) -> str:
        return ''

    def _case_default(self) -> str:
        return ''

    def _emit_case_start(self, pattern: str):
        pass

    def _emit_case_end(self):
        pass

    def _emit_if_start(self, sel: str):
        pass

    def _emit_elseif_start(self, sel: str):
        pass

    def _emit_else(self):
        pass

    def _emit_if_end(self):
        pass

    def _emit_submodule(self, submodule: rtlil.Cell):
        submodule.name = self.sanitize(submodule.name)
        submodule.kind = self.sanitize(submodule.kind)

        for port in tuple(submodule.ports.keys()):
            submodule.ports[self.sanitize(port)] = submodule.ports.pop(port)

    def _emit_operator(self, operator: rtlil.Cell, comb=True):
        lhs = operator.ports.get('Y', None)
        rhs = operator

        if lhs is None:
            raise RuntimeError("Operator without output!")

        with self._line.indent():
            self._emit_operator_assignment(rtlil.Assignment(lhs, rhs), comb=comb)

    def _operator_repr(self, operator: rtlil.Cell) -> str:
        return ''

    def _emit_signal(self, signal: rtlil.Wire):
        pass

    def _emit_memory(self, memory: Memory):
        processes, connections = memory.build(self._collect_signals, self.wire)

        for clk, process in processes:
            kwargs = {}
            if clk is not None:
                kwargs.update({'comb': False, 'clock': clk})
            self._emitted_processes.append((process, kwargs))

        self.connections.extend(connections)

    def _emit_module_definition(self):
        pass

    def _emit_flip_flop(self, flip_flop: rtlil.Cell):
        arst_value = self._convert_signals(flip_flop.parameters.get('ARST_VALUE', None))
        arst_polarity = flip_flop.parameters.get('ARST_POLARITY', True)
        arst = self._convert_signals(flip_flop.ports.get('ARST', None))
        data = self._convert_signals(flip_flop.ports['D'])
        out = self._convert_signals(flip_flop.ports['Q'])

        process = rtlil.Process(name=None)

        if arst is None:
            process.assign(out, data)
        else:
            if arst_value is None:
                raise RuntimeError("Missing arst value for async reset")

            switch = process.switch(arst)
            switch.case(['1' if arst_polarity else '0']).assign(out, arst_value)
            switch.default().assign(out, data)

        self._emit_process(
            process,
            comb = False,
            clock = flip_flop.ports['CLK'],
            polarity = flip_flop.parameters['CLK_POLARITY'],
            arst = arst,
            arst_polarity = arst_polarity,
        )

    def _emit_module_end(self):
        pass

    def _emit_connections(self):
        with self._line.indent():
            for lhs, rhs in self.connections:
                self._emit_assignment(rtlil.Assignment(lhs, rhs))

    def _get_initial(self, signal: rtlil.Wire) -> Union[str, list[str]]:
        ret: Union[_ast.Const, list[_ast.Const]] = signal.attributes.pop('init', None)
        if ret is not None:
            if isinstance(ret, list):
                ret = [self._const_repr(r.width, r.value) for r in ret]
            else:
                ret = self._const_repr(ret.width, ret.value)
        else:
            ret = self._const_repr(signal.width, 0)
        return ret

    @staticmethod
    def _const_repr(width, value):
        return ''

    def _collect_signals(self, signal: Union[str, _ast.Const], raw=False) -> Union[celosia_wire.Wire, list[celosia_wire.Wire]]:
        if raw:
            return self._get_raw_signals(signal)
        else:
            return self._convert_signals(signal)

    def _represent(self, signal: celosia_wire.Wire) -> str:
        if signal is None:
            return signal

        if isinstance(signal, celosia_wire.Concat):
            if len(signal.parts) == 1:
                return self._represent(signal.parts[0])
            return self._concat([self._represent(part) for part in signal.parts])

        if isinstance(signal, celosia_wire.Slice):
            wire_rep = self._represent(signal.wire)
            if signal.start_idx == 0 and signal.stop_idx >= signal.wire.width:
                return wire_rep
            return self._slice_repr(wire_rep, signal.start_idx, stop=signal.stop_idx-1)

        if isinstance(signal, (_ast.Const, celosia_wire.Const)):
            return self._const_repr(signal.width, signal.value)

        if isinstance(signal, celosia_wire.Cell):
            return self._operator_repr(signal.cell)

        if isinstance(signal, rtlil.Cell):
            return self._represent(celosia_wire.Cell(signal))

        if isinstance(signal, celosia_wire.MemoryIndex):
            return self._mem_slice_repr(signal)

        if isinstance(signal, celosia_wire.Wire):
            return signal.wire.name

        if isinstance(signal, rtlil.Wire):
            return signal.name

        # TODO: If everything else is handled correctly, this should not be needed
        if isinstance(signal, str):
            return self._represent(self._convert_signals(signal))

        raise ValueError(f"Unknown type to represent: {type(signal)}")

    def _convert_signals(self, signal: Any) -> Union[rtlil.Wire, rtlil.Cell]:
        if signal is None:
            return signal

        if isinstance(signal, celosia_wire.Wire):
            return signal

        # Wire
        if isinstance(signal, rtlil.Wire):
            return celosia_wire.Wire(signal)

        # Cell
        if isinstance(signal, rtlil.Cell):
            return celosia_wire.Cell(signal)

        # Const
        if isinstance(signal, _ast.Const):
            return celosia_wire.Const(signal.value, signal.width)

        if not isinstance(signal, str):
            raise RuntimeError(f"Unknown signal type: {type(signal)}")

        if signal.startswith('{') and signal.endswith('}'):
            signal = signal[1:-1]

        real_parts: list[celosia_wire.Wire] = []

        for part in signal.split():
            const = celosia_wire.Const.from_string(part)

            if const is not None:
                real_parts.append(const)

            elif part.startswith('[') and part.endswith(']'):
                part = part[1:-1]
                if ':' in part:
                    stop, start = map(int, part.split(':', maxsplit=1))
                else:
                    start = stop = int(part)

                wire = real_parts[-1]
                if wire.width != stop - start + 1:
                    real_parts[-1] = celosia_wire.Slice(wire, start_idx=start, stop_idx=stop+1)
            else:
                wire = self._signals.get(part, None)
                if wire is None:
                    raise RuntimeError(f"Unknown signal: {part}")
                real_parts.append(celosia_wire.Wire(wire))

        if len(real_parts) == 0:
            return celosia_wire.Const(0, 1) # TODO: Check

        elif len(real_parts) == 1:
            return real_parts[0]

        else:
            return celosia_wire.Concat(real_parts[::-1])

    def _get_raw_signals(self, signal: Any) -> list[Union[rtlil.Wire, rtlil.Cell]]:
        converted = self._convert_signals(signal)

        ret = []

        if isinstance(converted, celosia_wire.Const):
            pass
        elif isinstance(converted, celosia_wire.Slice):
            ret.append(converted.wire)
        elif isinstance(converted, celosia_wire.Concat):
            for part in converted.parts:
                ret.extend(self._get_raw_signals(part))
        elif isinstance(converted, celosia_wire.Cell):
            pass    # TODO: What do we do here?
        elif isinstance(converted, celosia_wire.Wire):
            ret.append(converted)
        else:
            raise RuntimeError(f"Unknown signal type: {type(signal)}")

        return ret

    @classmethod
    def _concat(cls, parts) -> str:
        return str(parts)

    @classmethod
    def _slice_repr(cls, name: str, start: int, stop: int = None) -> str:
        return ''

    def _mem_slice_repr(self, idx: celosia_wire.MemoryIndex):
        return self._slice_repr(idx.name, self._represent(idx.address))

    def _signed_division_fix(self, division: rtlil.Cell):
        DIVISION_OPERATORS = {
            "$divfloor",
            "$modfloor",
        }

        if not (
            (division.kind in DIVISION_OPERATORS) and
            (division.parameters['A_SIGNED'] or division.parameters['B_SIGNED'])
        ):
            self._emitted_operators.append(division)
            return

        operands = []
        for port in ('A', 'B'):
            operands.append(self._convert_signals(division.ports[port]))

        max_size = max(operand.width for operand in operands) + 2

        dividend = celosia_wire.Wire(self.wire(max_size))
        divisor = celosia_wire.Wire(self.wire(max_size))

        # for operand in operands:

        sign_bits0 = [operands[0][-1] for _ in range(max_size - operands[0].width)]
        sign_bits1 = [operands[1][-1] for _ in range(max_size - operands[1].width)]
        self.connect(dividend, celosia_wire.Concat([operands[0], *sign_bits0]))
        self.connect(divisor, celosia_wire.Concat([operands[1], *sign_bits1]))

        # dividend[-1] == divisor[-1]
        cmp_sign = self.wire(1)
        self.cell(kind = '$eq', name=None,
            ports = {
                'A': dividend[max_size-1],
                'B': divisor[max_size-1],
                'Y': cmp_sign,
            },
            parameters = {
                'A_WIDTH': 1,
                'A_SIGNED': False,
                'B_WIDTH': 1,
                'B_SIGNED': False,
            }
        )

        # dividend == 0
        cmp_zero = self.wire(1)
        self.cell(kind = '$eq', name=None,
            ports = {
                'A': dividend,
                'B': celosia_wire.Const(0, max_size),
                'Y': cmp_zero,
            },
            parameters = {
                'A_WIDTH': max_size,
                'A_SIGNED': True,
                'B_WIDTH': max_size,
                'B_SIGNED': True,
                'Y_WIDTH': 1,
            }
        )

        # divisor + 1
        addition = self.wire(max_size)
        self.cell(kind='$add', name=None,
            ports = {
                'A': divisor,
                'B': celosia_wire.Const(1, max_size),
                'Y': addition,
            },
            parameters = {
                'A_WIDTH': max_size,
                'A_SIGNED': True,
                'B_WIDTH': max_size,
                'B_SIGNED': True,
            }
        )

        # divisor - 1
        substraction = self.wire(max_size)
        self.cell(kind='$sub', name=None,
            ports = {
                'A': divisor,
                'B': celosia_wire.Const(1, max_size),
                'Y': substraction,
            },
            parameters = {
                'A_WIDTH': max_size,
                'A_SIGNED': True,
                'B_WIDTH': max_size,
                'B_SIGNED': True,
            }
        )

        # divisor[-1] ? (divisor + 1) : (divisor - 1)
        substraction_mux = self.wire(max_size)
        self.cell(kind = '$mux', name = None,
            ports = {
                'S': divisor[-1],
                'A': substraction,
                'B': addition,
                'Y': substraction_mux,
            },
        )

        # dividend - (divisor[-1] ? (divisor + 1) : (divisor - 1))
        mux_false_operand = self.wire(max_size)
        self.cell(kind='$sub', name=None,
            ports = {
                'A': dividend,
                'B': substraction_mux,
                'Y': mux_false_operand,
            },
            parameters = {
                'A_WIDTH': max_size,
                'A_SIGNED': True,
                'B_WIDTH': max_size,
                'B_SIGNED': True,
            }
        )

        # (dividend[-1] == divisor[-1]) | (dividend == 0)
        mux_sel = self.wire(1)
        self.cell(kind = '$or', name=None,
            ports = {
                'A': cmp_sign,
                'B': cmp_zero,
                'Y': mux_sel,
            },
            parameters = {
                'A_WIDTH': 1,
                'A_SIGNED': False,
                'B_WIDTH': 1,
                'B_SIGNED': False,
                'Y_WIDTH': 1,
            }
        )

        # (dividend[-1] == divisor[-1]) | (dividend == 0) ? dividend : (dividend - (divisor[-1] ? (divisor + 1) : (divisor - 1)))
        real_dividend = self.wire(max_size)
        self.cell(kind = '$mux', name = None,
            ports = {
                'S': mux_sel,
                'A': mux_false_operand,
                'B': dividend,
                'Y': real_dividend,
            },
        )

        self._emitted_operators.append(rtlil.Cell(kind = division.kind, name = None,
            ports = {
                'A': celosia_wire.Wire(real_dividend),
                'B': divisor,
                'Y': division.ports['Y'],
            },
            parameters = {
                'A_WIDTH': max_size,
                'A_SIGNED': True,
                'B_WIDTH': max_size,
                'B_SIGNED': True,
                'Y_WIDTH': division.parameters['Y_WIDTH'],
            }
        ))

    def _signed(self, value) -> str:
        return ''
