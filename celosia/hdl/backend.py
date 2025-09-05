from amaranth.back import rtlil
from amaranth.back.rtlil import _const
from celosia.hdl.memory import Memory
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
        self._emitted_processes: list[rtlil.Process] = []
        self._emitted_submodules: list[rtlil.Cell] = []
        self._emitted_signals: list[rtlil.Wire] = []
        self._emitted_memories: dict[str, Memory] = {}
        self._emitted_operators: list[rtlil.Cell] = []
        self._emitted_flip_flops: list[rtlil.Cell] = []

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

    def emit(self, line: rtlil.Emitter):
        self._line = line

        line.port_id = 0

        for name, cell in self.contents.items():
            destination = None
            ignore = False

            if isinstance(cell, rtlil.Wire):
                if cell.port_kind is None:
                    destination = self._emitted_signals
                else:
                    destination = self._emitted_ports
            elif isinstance(cell, rtlil.Process):
                destination = self._emitted_processes
            elif isinstance(cell, rtlil.Cell):
                if self._cell_is_submodule(cell):
                    destination = self._emitted_submodules
                elif self._cell_is_ff(cell):
                    destination = self._emitted_flip_flops
                elif self._cell_is_memory(cell):
                    self._get_memory_from_port(cell).set_cell(cell)
                    ignore = True
                elif self._cell_is_memory_wp(cell):
                    self._get_memory_from_port(cell).add_wp(cell)
                    ignore = True
                elif self._cell_is_memory_rp(cell):
                    self._get_memory_from_port(cell).add_rp(cell)
                    ignore = True
                elif self._cell_is_yosys(cell):
                    destination = self._emitted_operators
            elif isinstance(cell, rtlil.Memory):
                self._emitted_memories[cell.name] = Memory(cell.name, cell, self._signals, self._collect_signals)
                ignore = True

            if not ignore:
                if destination is None:
                    raise RuntimeError(f"Unknown cell type named {name}: {type(cell)}")
                destination.append(cell)

        self._emit_module_and_ports(self._emitted_ports)

        for signal in self._emitted_signals:
            self._emit_signal(signal)

        for memory in self._emitted_memories.values():
            self._emit_memory(memory)

        for flip_flop in self._emitted_flip_flops:
            self._emit_flip_flop(flip_flop)

        for process in self._emitted_processes:
            self._emit_process(process)

        for operator in self._emitted_operators:
            self._emit_operator(operator)

        for submodule in self._emitted_submodules:
            self._emit_submodule(submodule)

        self._emit_connections()
        self._emit_module_end()

        line()

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

    def _emit_switch(self, switch: rtlil.Switch, comb=True):
        with self._line.indent():
            self._emit_switch_start(self._get_signal_name(switch.sel))

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
        print('Emit submodule:', submodule.name, submodule.kind)
        pass

    def _emit_operator(self, operator: rtlil.Cell):
        print('Emit operator:', operator.name, operator.kind, operator.ports)
        pass

    def _emit_signal(self, signal: rtlil.Wire):
        self._signals[signal.name] = signal

    def _emit_memory(self, memory: Memory):
        memory.build()

    def _emit_module_and_ports(self, ports: list["rtlil.Wire"]):
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
        for lhs, rhs in self.connections:
            self._emit_assignment(rtlil.Assignment(lhs, rhs))

    def _get_initial(self, signal: rtlil.Wire) -> str:
        ret: _ast.Const = signal.attributes.pop('init', None)
        if ret is not None:
            ret = self._const_repr(ret.width, ret.value)
        return ret

    @staticmethod
    def _const_repr(width, value):
        return ''

    def _get_signal_name(self, signal: Union[str, _ast.Const]) -> str:
        ret = self._collect_signals(signal, raw=False)
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
    def _concat(cls, parts):
        return str(parts)

    @classmethod
    def _get_slice(cls, name: str, start: int, stop: int):
        return ''
