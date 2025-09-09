from amaranth.back import rtlil
from amaranth.hdl import _ast
from typing import Union

def _get_slice_params(slice: slice, width: int):
    start = slice.start
    if start is None:
        start = 0
    if start < 0:
        start += width
    # start = min(width, max(0, start))
    start = max(0, start)

    stop = slice.stop
    if stop is None:
        stop = width
    if stop < 0:
        stop += width
    stop = min(width, stop)

    return start, stop

class Wire:
    def __init__(self, wire: rtlil.Wire):
        self.wire = wire

        self.name = wire.name
        self.width = wire.width

    def __getitem__(self, index):
        if isinstance(index, int):
            return Slice(self, index)
        if isinstance(index, slice):
            start, stop = _get_slice_params(index, self.width)
            if stop - start >= self.width:
                return self
            else:
                return Slice(self, start, stop)
        raise ValueError(f"Unknown index: {index}")

    def __eq__(self, value):
        if isinstance(value, Slice):
            return (self == value.wire) and (value.start_idx <= 0) and (value.stop_idx >= self.width)
        if isinstance(value, Concat):
            return (len(value.parts) == 1) and (self == value.parts[0])
        if isinstance(value, Wire):
            return self.name == value.name
        return False

class Cell(Wire):
    def __init__(self, cell: rtlil.Cell):
        self.cell = cell
        self.width = cell.parameters.get('Y_WIDTH', None) or cell.parameters.get('WIDTH', None)

class Slice(Wire):
    def __init__(self, wire: Wire, start_idx=None, stop_idx=None):
        self.wire = wire
        start, stop = _get_slice_params(slice(start_idx, stop_idx), wire.width)
        self.start_idx = start
        self.stop_idx = stop
        self.width = stop - start

    def __getitem__(self, index):
        if isinstance(index, int):
            index, _ = _get_slice_params(slice(index, index+1), self.width)
            return Slice(self.wire, self.start_idx + index, self.stop_idx)
        if isinstance(index, slice):
            _start, _stop = _get_slice_params(index, self.width)

            start = self.start_idx + _start
            stop  = min(start + self.stop_idx, _stop)

            if stop - start >= self.wire.width:
                return self.wire
            else:
                return Slice(self.wire, start, stop)
        raise ValueError(f"Unknown index: {index}")

    def __eq__(self, value):
        if isinstance(value, Slice):
            return (self.wire == value.wire) and (value.start_idx == self.start_idx) and (value.stop_idx == self.stop_idx)
        if isinstance(value, Concat):
            return (len(value.parts) == 1) and (self == value.parts[0])
        if isinstance(value, Wire):
            return (value == self)
        return False

class Concat(Wire):
    def __init__(self, wires: list[Wire]):
        self.parts = self._cleanup(wires)
        self.width = sum(wire.width for wire in wires)

    @classmethod
    def _cleanup(cls, wires):
        real_wires = []
        curr_value = curr_width = 0

        def emit_const():
            if curr_width:
                real_wires.append(Const(curr_value, curr_width))
            return 0, 0

        for wire in wires:
            if isinstance(wire, Const):
                value = wire.value
                if value < 0:
                    value += 2**wire.width
                curr_value |= (value << curr_width)
                curr_width += wire.width
            else:
                curr_value, curr_width = emit_const()
                if isinstance(wire, Concat):
                    real_wires.extend(wire.parts)
                else:
                    real_wires.append(wire)

        curr_value, curr_width = emit_const()
        return real_wires

    def __getitem__(self, index):
        if isinstance(index, int):
            index, _ = _get_slice_params(slice(index, index+1), self.width)
            curr_idx = 0
            for part in self.parts:
                if curr_idx + part.width > index:
                    return part[index - curr_idx]
                curr_idx += part.width
            return Const(0, 1)

        if isinstance(index, slice):
            start, stop = _get_slice_params(index, self.width)

            included = []
            curr_idx = 0

            for part in self.parts:
                if curr_idx >= stop:
                    break

                if curr_idx + part.width < start:
                    curr_idx += part.width
                    continue

                start_idx = max(0, start - curr_idx)
                included.append(part[start_idx : start_idx + min(stop - curr_idx, part.width)])
                curr_idx += part.width

            if len(included) == 0:
                return Const(0, stop - start)
            elif len(included) == 1:
                return included[0]
            else:
                return Concat(included)

        raise ValueError(f"Unknown index: {index}")

    def __eq__(self, value):
        if isinstance(value, Slice):
            return (value == self)
        if isinstance(value, Concat):
            return (len(value.parts) == len(self.parts)) and all(other_p == self_p for other_p, self_p in zip(value.parts, self.parts))
        if isinstance(value, Wire):
            return (value == self)
        return False

class Const(Wire):
    def __init__(self, value: int, width: int):
        self.value = value
        self.width = width

    def __getitem__(self, index):
        if isinstance(index, int):
            index, _ = _get_slice_params(slice(index, index+1), self.width)
            return Const((self.value >> index) & 1, 1)
        if isinstance(index, slice):
            start, stop = _get_slice_params(index, self.width)
            width = stop - start
            return Const((self.value >> start) & int('1' * width, 2), width)
        raise ValueError(f"Unknown index: {index}")

    def __eq__(self, value):
        if isinstance(value, Const):
            return (self.value == value.value)

        if isinstance(value, Concat):
            return (len(value.parts) == 1) and (value.parts[0] == self)
        return False

    @classmethod
    def from_string(cls, string: str) -> "Const":
        if string is None:
            return None

        width_idx = string.find("'")
        if width_idx < 0:
            return None

        width = int(string[:width_idx])
        string = string[width_idx+1:]

        if len(string) != width:
            raise ValueError(f"Inconsistent width for const string: {string} ({width})")

        # FIX: Only for case patterns, should never be converted to int
        if '-' in string:
            return cls(string, width)

        return cls(int(string, 2), width)

class MemoryIndex(Wire):
    def __init__(self, memory: rtlil.Memory, address: Wire):
        self.memory = memory

        self.name = memory.name
        self.width = memory.width

        self.address = address

    def __eq__(self, value):
        if isinstance(value, MemoryIndex):
            return (value.memory is self.memory) and (value.address == self.address)
        return False