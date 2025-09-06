from amaranth.back import rtlil
from amaranth.back.rtlil import _const
from celosia.hdl.memory import Memory, MemoryIndex
from typing import Any, Union
from amaranth.hdl import _ast
import re

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

        self.name = self._sanitize(self.name)

        self._process_id = 0

    @classmethod
    def _const(cls, value: Any):
        return _const(value)

    def _sanitize(self, name: str) -> str:
        return name

    def _auto_name(self):
        return self._sanitize(super()._auto_name())

    def _name(self, name):
        ret = super()._name(name)
        if ret.startswith('\\'):
            ret = ret[1:]
        return ret

    def wire(self, *args, **kwargs):
        wire = super().wire(*args, **kwargs)
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
        if self._cell_is_submodule(cell):
            self._emitted_submodules.append(cell)
        elif self._cell_is_ff(cell):
            self._emitted_flip_flops.append(cell)
        elif self._cell_is_memory(cell):
            self._get_memory_from_port(cell).set_cell(cell)
        elif self._cell_is_memory_wp(cell):
            self._get_memory_from_port(cell).add_wp(cell)
        elif self._cell_is_memory_rp(cell):
            self._get_memory_from_port(cell).add_rp(cell)
        # elif self._cell_is_yosys(cell):
        else:
            self._emitted_operators.append(cell)
        return cell

    def memory(self, *args, **kwargs):
        memory = super().memory(*args, **kwargs)
        self._emitted_memories[memory.name] = self._signals[memory.name] = Memory(memory)
        return memory

    def emit(self, line: rtlil.Emitter):
        self._line = line
        line.port_id = 0

        # Memories need to go first because they may create new signals, processes and connections
        for memory in self._emitted_memories.values():
            self._emit_memory(memory)

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
        return not cls._cell_is_yosys(cell)

    @classmethod
    def _cell_is_ff(cls, cell: rtlil.Cell):
        return cls._cell_is_yosys(cell) and cell.kind in ('$dff', '$adff')

    @classmethod
    def _cell_is_memory(cls, cell: rtlil.Cell):
        return cls._cell_is_yosys(cell) and cell.kind == '$meminit_v2'

    @classmethod
    def _cell_is_memory_wp(cls, cell: rtlil.Cell):
        return cls._cell_is_yosys(cell) and cell.kind == '$memwr_v2'

    @classmethod
    def _cell_is_memory_rp(cls, cell: rtlil.Cell):
        return cls._cell_is_yosys(cell) and cell.kind == '$memrd_v2'

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

    @classmethod
    def _is_switch_if(cls, switch: rtlil.Switch) -> bool:
        used_idx: set[int] = set()
        for case in switch.cases:
            if not case.patterns:
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
        if self._is_switch_if(switch):
            # return self._emit_if(switch, comb=comb)
            pass

        # TODO: Handle operators (mostly for memory transparency, maybe it can be a special case)
        # self._emit_switch_start(self._get_signal_name(switch.sel))
        self._emit_switch_start(switch.sel)

        with self._line.indent():
            for case in switch.cases:
                if case.patterns:
                    pattern = self._case_patterns((self._get_signal_name(f"{len(pattern)}'{pattern}") for pattern in case.patterns))
                else:
                    pattern = self._case_default()

                self._emit_case_start(pattern)
                with self._line.indent():
                    self._emit_process_contents(case.contents, comb=comb)
                self._emit_case_end()

        self._emit_switch_end()

    def _emit_if(self, switch: rtlil.Switch, comb=True):
        pass

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

    def _emit_submodule(self, submodule: rtlil.Cell):
        submodule.name = self._sanitize(submodule.name)
        submodule.kind = self._sanitize(submodule.kind)

    def _emit_operator(self, operator: rtlil.Cell):
        print('Emit operator:', operator.name, operator.kind, operator.ports)
        pass

    def _emit_signal(self, signal: rtlil.Wire):
        pass

    def _emit_memory(self, memory: Memory):
        processes, connections = memory.build(self._signals, self._collect_signals, self.wire)

        for clk, process in processes:
            kwargs = {}
            if clk is not None:
                kwargs.update({'comb': False, 'clock': clk})
            self._emitted_processes.append((process, kwargs))

        self.connections.extend(connections)

    def _emit_module_definition(self):
        pass

    def _emit_flip_flop(self, flip_flop: rtlil.Cell):
        arst_value = self._get_signal_name(flip_flop.parameters.get('ARST_VALUE', None))
        arst_polarity = flip_flop.parameters.get('ARST_POLARITY', True)
        arst = self._get_signal_name(flip_flop.ports.get('ARST', None))
        data = self._get_signal_name(flip_flop.ports['D'])
        out = self._get_signal_name(flip_flop.ports['Q'])

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
            clock = self._get_signal_name(flip_flop.ports['CLK']),
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
        return ret

    @staticmethod
    def _const_repr(width, value):
        return ''

    def _get_signal_name(self, signal: Union[str, _ast.Const]) -> str:
        ret = self._collect_signals(signal, raw=False)

        if ret is None:
            return ret

        if len(ret) == 1:
            return ret[0]
        else:
            return self._concat(ret)

    def _get_raw_signals(self, signal: Union[str, _ast.Const]) -> list[str]:
        return self._collect_signals(signal, raw=True)

    def _collect_signals(self, signal: Union[str, _ast.Const], raw=False) -> list[str]:
        if signal is None:
            return None

        if isinstance(signal, _ast.Const):
            return [] if raw else [self._const_repr(signal.width, signal.value)]

        if isinstance(signal, MemoryIndex):
            indexed_mem = self._get_mem_slice(signal)
            if raw or signal.slice is None:
                return [indexed_mem]
            else:
                return [self._get_slice(indexed_mem, signal.slice.start, signal.slice.stop - 1)]

        const_pattern = re.compile(r"(\d+)'([\d|-]+)")
        slice_pattern = re.compile(r'(.*?) \[(.*?)\]')

        if signal.startswith('{') and signal.endswith('}'):
            signal = signal[1:-1].strip()
            concat = True
        else:
            concat = False

        real_parts = []

        while signal:
            const_match = const_pattern.match(signal)

            if const_match is not None:
                if not raw:
                    real_parts.append(self._const_repr(*const_match.groups()))
                signal = signal[const_match.end() + 1:]
                continue

            slice_match = slice_pattern.match(signal)
            if slice_match is not None:
                name = slice_match.group(1)

                if raw:
                    real_parts.append(name)

                else:
                    wire = self._signals.get(name, None)
                    if wire is None:
                        raise RuntimeError(f"Unknown signal: {name}")

                    index = slice_match.group(2)

                    if ':' in index:
                        stop, start = map(int, index.split(':'))
                    else:
                        stop = start = int(index)

                    if wire.width == stop - start + 1:
                        real_parts.append(name)
                    else:
                        real_parts.append(f'{self._get_slice(name, start, stop)}')

                signal = signal[slice_match.end() + 1:]
                continue

            space_idx = signal.find(' ')
            if space_idx < 0:
                space_idx = len(signal)

            real_parts.append(signal[:space_idx])
            signal = signal[space_idx + 1:]

        return real_parts

    @classmethod
    def _concat(cls, parts) -> str:
        return str(parts)

    @classmethod
    def _get_slice(cls, name: str, start: int, stop: int) -> str:
        return ''

    @classmethod
    def _get_mem_slice(cls, idx: MemoryIndex):
        return cls._get_slice(idx.name, idx.address, idx.address)
