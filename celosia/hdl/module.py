from amaranth.back import rtlil
from amaranth.back.rtlil import _const
from celosia.hdl.memory import Memory
import celosia.hdl.wire as celosia_wire
from typing import Any, Union
from amaranth.hdl import _ast

# TODO: Tap into rtlil.ModuleEmitter so we can have control over Wire names

class Module(rtlil.Module):

    submodules_first = False
    case_sensitive = True
    _DEFERRED_WIRE_MARKER = '$____celosia_internal_deferred$'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._line: rtlil.Emitter = None

        self._signals: dict[str, rtlil.Wire] = {}

        self._emitted_ports: list[rtlil.Wire] = []
        self._emitted_processes: list[tuple[rtlil.Process, dict]] = []
        self._emitted_submodules: list[rtlil.Cell] = []
        self._emitted_memories: dict[str, Memory] = {}
        self._emitted_operators: list[rtlil.Cell] = []

        self._process_wires: set[str] = set()

        self.name = self.sanitize(self.name)
        self._assigned_names: set[str] = set()
        self._renamed: set[str] = set()

        self._rp_count = 0

        self._SANITIZED_DEFERRED_MARKED = self.sanitize(self._DEFERRED_WIRE_MARKER)

    @classmethod
    def _const(cls, value: Any):
        return _const(value)

    @classmethod
    def sanitize(cls, name: str) -> str:
        return name

    def _auto_name(self):
        return self._filter_name(super()._auto_name())

    def _name(self, name):
        ret = super()._name(name)
        if ret.startswith('\\'):
            ret = self._filter_name(ret[1:])
        return ret

    @classmethod
    def _change_case(cls, name: str) -> str:
        return name if cls.case_sensitive else name.lower()

    @classmethod
    def filter_name(self, name: str, assigned_names: set[str], change_case = True) -> str:
        name = self.sanitize(name)

        if change_case:
            assigned_names = {self._change_case(name) for name in assigned_names}

        curr_num = ''
        curr_idx = len(name) - 1
        while curr_idx >= 0 and name[curr_idx].isnumeric():
            curr_num = name[curr_idx] + curr_num
            curr_idx -= 1

        if curr_num:
            idx = int(curr_num) + 1
            _name = name[:curr_idx+1]
        else:
            idx = 0
            _name = name

        while self._change_case(name) in assigned_names:
            name = f'{_name}{idx}'
            idx += 1

        return name

    def _filter_name(self, name: str) -> str:
        name = self.filter_name(name, assigned_names=self._assigned_names, change_case=False)
        self._assigned_names.add(self._change_case(name))
        return name

    def _anonymous_name(self, name: str) -> str:
        if name is None:
            # We'll rename this later
            name = self._DEFERRED_WIRE_MARKER
        return name

    def wire(self, *args, name=None, **kwargs):
        wire = super().wire(*args, name=self._anonymous_name(name), **kwargs)
        self._signals[wire.name] = wire
        if wire.port_kind is not None:
            self._emitted_ports.append(wire)
        return wire

    def process(self, *args, name=None, **kwargs):
        process = super().process(*args, name=self._anonymous_name(name), **kwargs)
        self._emitted_processes.append((process, {}))
        return process

    def cell(self, *args, **kwargs):
        cell = super().cell(*args, **kwargs)

        rename_output = False

        if self._cell_is_submodule(cell):
            self._emitted_submodules.append(cell)
        elif self._cell_is_ff(cell):
            self._emit_flip_flop(cell)
        elif self._cell_is_memory(cell):
            self._get_memory_from_port(cell).set_cell(cell)
        elif self._cell_is_memory_wp(cell):
            self._get_memory_from_port(cell).add_wp(cell)
        elif self._cell_is_memory_rp(cell):
            if self._get_memory_from_port(cell).add_rp(cell, self._rp_count).clk_enable:
                self._rp_count += 1
        else:
            rename_output = True
            if self._cell_is_division(cell):
                cell = self._signed_division_fix(cell)
            self._emitted_operators.append(cell)

        self._convert_ports(cell, rename_output=rename_output)
        return cell

    def memory(self, *args, **kwargs):
        memory = super().memory(*args, **kwargs)
        self._emitted_memories[memory.name] = self._signals[memory.name] = Memory(memory)
        return memory

    def _convert_ports(self, cell: rtlil.Cell, rename_output=False):
        if rename_output:
            output = self._convert_signals(cell.ports.get('Y', None))
            self._rename_anonymous(output, cell=cell)

        for key, value in cell.ports.items():
            converted = self._convert_signals(value)
            for signal in self._get_raw_signals(converted):
                self._rename_anonymous(signal, kind='i')
            cell.ports[key] = converted

    def _rename_anonymous(self, wire: Union[celosia_wire.Component, Any], **kwargs):
        if not hasattr(wire, 'name'):
            return

        old_name = wire.name
        if old_name is None or (isinstance(old_name, str) and old_name.startswith(self._SANITIZED_DEFERRED_MARKED)):
            new_name = self._get_name_for_cell_output(**kwargs)
            wire.name = new_name

            if isinstance(wire, celosia_wire.Wire):
                # self._signals.pop(wire.name, None)
                self._renamed.add(old_name)#] = new_name
                self._signals[new_name] = wire.wire

    # TODO: Reactive when cell names are improved
    # def _get_name_for_cell_input(self, port: celosia_wire.Component) -> str:
    #     ret = None
    #     if isinstance(port, celosia_wire.Const):
    #         ret = str(port.value)
    #     elif isinstance(port, celosia_wire.Slice):
    #         internal = self._get_name_for_cell_input(port.wire)
    #         if internal is not None:
    #             ret = f'{internal}_sliced'
    #     elif isinstance(port, celosia_wire.Concat):
    #         if len(port.parts) == 1:
    #             ret = self._get_name_for_cell_input(port.parts[0])
    #         else:
    #             ret = 'concat'
    #     elif isinstance(port, celosia_wire.Cell):
    #         ret = 'cell'    # TODO: Improve?
    #     elif isinstance(port, (celosia_wire.Wire, celosia_wire.MemoryIndex)):
    #         ret = port.name

    #     return ret

    def _get_name_for_cell_output(self, cell: Union[rtlil.Cell, Any] = None, kind: str = None, prefix: Any = None) -> str:
        op_names = {
            '$neg': 'negative', '$sub': 'subtraction', '$not': 'inverted', '$add': 'addition', '$mul': 'multiplication',
            '$and': 'bitwise_and', '$xor': 'bitwise_xor', '$or': 'bitwise_or', '$divfloor': 'division', '$modfloor': 'remanent',
            '$shl': 'lshift', '$shr': 'rshift', '$shift': 'rshift', '$sshr': 'rshift',
            '$reduce_bool': 'reduce_bool', '$reduce_and': 'reduce_and', '$reduce_xor': 'reduce_xor', '$reduce_or': 'reduce_or',
            '$eq': 'equals', '$ne': 'notequals', '$lt': 'less_than', '$gt': 'greater_than', '$le': 'less_or_equals',
            '$ge': 'greater_or_equals', '$mux': 'mux',

            'p': 'proc',        # Internal, for processes
            'i': 'internal',    # Internal, for intermediate signals
            'f': '_next',       # Internal, for flip-flops
        }

        if kind is None:
            kind = getattr(cell, 'kind', None)

        final_name = op_name = None

        if kind is not None:
            op_name = op_names.get(kind, None)

        if op_name is not None:
            final_name = op_name

        if hasattr(prefix, 'name'):
            prefix = prefix.name

        if isinstance(prefix, str):
            final_name = f'{prefix}{final_name}'

        return self._name(final_name)

        # TODO: Improved naming for operations! It works, but names get weird quickly
        # op_names = {
        #     '$neg': 'negative', '$sub': 'minus', '$not': 'inv', '$add': 'plus', '$mul': 'times', '$and': 'and',
        #     '$xor': 'xor', '$or': 'or', '$divfloor': 'div', '$modfloor': 'rem',
        #     '$shl': 'lshifted', '$shr': 'rshifted', '$shift': 'rshifted', '$sshr': 'rshifted',
        #     '$reduce_bool': 'reduce_bool', '$reduce_and': 'reduce_and', '$reduce_xor': 'reduce_xor', '$reduce_or': 'reduce_or',
        #     '$eq': 'equals', '$ne': 'notequals', '$lt': 'less_than', '$gt': 'greater_than', '$le': 'less_or_equals',
        #     '$ge': 'greater_or_equals', '$mux': 'mux',
        # }

        # names = []

        # if cell.kind in op_names:
        #     converted: list[celosia_wire.Component] = []
        #     if cell.kind == '$mux':
        #         converted.append(self._convert_signals(cell.ports['S']))

        #     converted = [self._convert_signals(cell.ports['A'])]
        #     port_b = cell.ports.get('B', None)
        #     if port_b is not None:
        #         converted.append(self._convert_signals(cell.ports['B']))

        #     names.extend(
        #         self._get_name_for_cell_input(port) for port in converted
        #     )

        # if names and all(names):
        #     ret = f'{names[0]}_{op_names[cell.kind]}'
        #     for name in names[1:]:
        #         ret += f'_{name}'
        #     if ret[-1].isnumeric():
        #         ret += '_'
        #     ret = self._name(ret)
        # else:
        #     ret = self._auto_name()

        # return ret

    def _collect_lhs(self, assignment: Any) -> set[str]:
        ret = set()

        # TODO: We can probably break early if LHS is never a concatenation
        if isinstance(assignment, rtlil.Assignment):
            ret.update(wire.name for wire in self._get_raw_signals(assignment.lhs))

        elif isinstance(assignment, rtlil.Switch):
            for case in assignment.cases:
                ret.update(self._collect_lhs(case))

        elif isinstance(assignment, (rtlil.Case, rtlil.Process)):
            for content in assignment.contents:
                ret.update(self._collect_lhs(content))

        return ret

    def emit(self, line: rtlil.Emitter):
        self._line = line
        line.port_id = 0

        # Memories need to go first because they may create new signals, processes and connections
        for memory in self._emitted_memories.values():
            self._emit_memory(memory)

        for process, _ in self._emitted_processes:
            self._rename_anonymous(process, kind='p')
            for name in self._collect_lhs(process):
                wire = self._signals[name]
                self._rename_anonymous(celosia_wire.Wire(wire), kind='i')
                self._process_wires.add(wire.name)

        # After this point, no new signals, processes or connections are created
        # We have this callback so subclasses can gather all the information they may need before starting
        self._emit_pre_callback()

        self._emit_module_definition()
        for name, signal in self._signals.items():
            if name not in self._renamed:
                self._emit_signal(signal)

        # After this point, all signals have been emitted
        # We have this callback so subclasses can gather all the information they may need before starting
        self._emit_post_callback()

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

    def _emit_post_callback(self):
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

    def _emit_process(self, process: rtlil.Process, comb = True, **kwargs):
        with self._line.indent():
            self._emit_process_start(process.name, **kwargs)

            with self._line.indent():
                self._emit_process_contents(process.contents, comb=comb)

            self._emit_process_end(process.name, comb=comb)

    def _emit_process_start(self, name: str, clock: str = None, polarity: bool = True, arst: str = None, arst_polarity = False):
        pass

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

    def _strip_unused_cases(self, switch: rtlil.Switch):
        patterns = []
        bits = self._convert_signals(switch.sel).width
        for idx in reversed(range(len(switch.cases))):
            case = switch.cases[idx]
            case_patterns = case.patterns
            if not case_patterns:
                case_patterns = ('-' * bits,)

            if not case.contents:
                for pattern in patterns:
                    if all(p == c or p == '-' or c == '-' for case_pattern in case_patterns for p, c in zip(pattern, case_pattern)):
                        break
                else:
                    switch.cases.pop(idx)
            else:
                patterns.extend(case_patterns)

    def _emit_switch(self, switch: rtlil.Switch, comb=True, ignore_unused=False):
        if not ignore_unused:
            self._strip_unused_cases(switch)

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
        sel = self._convert_signals(switch.sel)
        first = True

        for case in switch.cases:
            if case.patterns:
                assert len(case.patterns) == 1, "Internal error" # Should never happen
                condition = self._represent(sel[case.patterns[0][::-1].index('1')], boolean=True)
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
        instance = '.' not in submodule.kind
        submodule.kind = self.sanitize(submodule.kind)

        # Shouldn't need to sanitize!
        # submodule.name = self.sanitize(submodule.name)

        # for port in tuple(submodule.ports.keys()):
        #     submodule.ports[self.sanitize(port)] = submodule.ports.pop(port)

        self._emit_submodule_post(submodule, instance = instance)

    def _emit_submodule_post(self, submodule: rtlil.Cell, instance: bool):
        pass

    def _emit_operator(self, operator: rtlil.Cell, comb=True):
        lhs = operator.ports.get('Y', None)
        rhs = operator

        if lhs is None:
            raise RuntimeError("Operator without output!")

        with self._line.indent():
            self._emit_operator_assignment(rtlil.Assignment(lhs, rhs), comb=comb)

    def _operator_repr(self, operator: rtlil.Cell, boolean: bool = False) -> str:
        return ''

    def _emit_signal(self, signal: rtlil.Wire):
        pass

    def _emit_memory(self, memory: Memory):
        processes, connections = memory.build(self.wire)

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

        self._rename_anonymous(data, kind='f', prefix=out)

        process = self.process()

        if arst is None:
            process.assign(out, data)
        else:
            if arst_value is None:
                raise RuntimeError("Missing arst value for async reset")

            switch = process.switch(arst)
            switch.case(['1' if arst_polarity else '0']).assign(out, arst_value)
            switch.default().assign(out, data)

        self._emitted_processes[-1] = (process, {
            'comb':  False,
            'clock':  flip_flop.ports['CLK'],
            'polarity':  flip_flop.parameters['CLK_POLARITY'],
            'arst':  arst,
            'arst_polarity':  arst_polarity,
        })

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
                ret = [self._const_repr(r.width, r.value, init=True) for r in ret]
            else:
                ret = self._const_repr(ret.width, ret.value, init=True)
        else:
            ret = self._const_repr(signal.width, 0, init=True)
        return ret

    @staticmethod
    def _const_repr(width, value, init=False):
        return ''

    def _represent(self, signal: celosia_wire.Component, boolean = False) -> str:
        if signal is None:
            return signal

        if isinstance(signal, celosia_wire.Concat):
            parts_repr = [self._represent(part, boolean=boolean) for part in signal.parts]
            if len(signal.parts) == 1:
                return parts_repr[0]
            return self._concat(parts_repr)

        if isinstance(signal, celosia_wire.Slice):
            if boolean:
                return self._to_boolean(signal)
            else:
                wire_rep = self._represent(signal.wire, boolean=False)
                if signal.start_idx == 0 and signal.stop_idx >= signal.wire.width:
                    return wire_rep
                return self._slice_repr(wire_rep, signal.start_idx, stop=signal.stop_idx-1)

        if isinstance(signal, (_ast.Const, celosia_wire.Const)):
            if boolean:
                return self._to_boolean(celosia_wire.Const(signal.value, signal.width))
            return self._const_repr(signal.width, signal.value)

        if isinstance(signal, celosia_wire.Cell):
            return self._operator_repr(signal.cell, boolean=boolean)

        if isinstance(signal, rtlil.Cell):
            return self._represent(celosia_wire.Cell(signal), boolean=boolean)

        if isinstance(signal, celosia_wire.MemoryIndex):
            return self._mem_slice_repr(signal)

        if isinstance(signal, celosia_wire.Wire):
            if boolean:
                return self._to_boolean(signal)
            return signal.wire.name

        if isinstance(signal, rtlil.Wire):
            if boolean:
                return self._to_boolean(celosia_wire.Wire(signal))
            return signal.name

        # TODO: If everything else is handled correctly, this should not be needed
        if isinstance(signal, str):
            return self._represent(self._convert_signals(signal), boolean=boolean)

        raise ValueError(f"Unknown type to represent: {type(signal)}")

    def _to_boolean(self, signal: celosia_wire.Component):
        return self._represent(signal, boolean=False)

    def _convert_signals(self, signal: Any) -> celosia_wire.Component:
        if signal is None:
            return signal

        if isinstance(signal, celosia_wire.Component):
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

        real_parts: list[celosia_wire.Component] = []

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
        elif isinstance(converted, (celosia_wire.Wire, celosia_wire.MemoryIndex)):
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
        if not (
            (division.parameters['A_SIGNED'] or division.parameters['B_SIGNED'])
        ):
            # FIX: Initialize to non-zero for simulation
            for wire in self._get_raw_signals(division.ports['B']):
                if isinstance(wire, celosia_wire.Wire):
                    if 'init' not in wire.wire.attributes:
                        wire.wire.attributes['init'] = _ast.Const(1, wire.wire.width)

            return division

        operands = []
        for port in ('A', 'B'):
            operands.append(self._convert_signals(division.ports[port]))

        max_size = max(operand.width for operand in operands) + 2

        dividend = celosia_wire.Wire(self.wire(max_size))
        # FIX: Initialize to non-zero for simulation
        divisor = celosia_wire.Wire(self.wire(max_size, attrs={'init': celosia_wire.Const(1, max_size)}))

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
                'Y_WIDTH': 1,
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
                'Y_WIDTH': max_size,
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
                'Y_WIDTH': max_size,
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
            parameters = {
                'WIDTH': max_size,
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
                'Y_WIDTH': max_size,
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
            parameters = {
                'WIDTH': max_size,
            },
        )

        return rtlil.Cell(kind = division.kind, name = None,
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
        )

    def _signed(self, value) -> str:
        return ''
