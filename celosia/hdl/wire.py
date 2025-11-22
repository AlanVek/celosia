from amaranth.back import rtlil
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

class Component:
    @property
    def width(self) -> int:
        raise NotImplementedError("Component width must be overwritten!")

    def __len__(self) -> int:
        return self.width

class Wire(Component):
    def __init__(self, wire: Union[rtlil.Wire, "Wire"]):
        if isinstance(wire, Wire):
            self.wire = wire.wire
        elif isinstance(wire, rtlil.Wire):
            self.wire = wire
        else:
            raise RuntimeError(f"Invalid wire: {wire}")

    @property
    def name(self) -> str:
        return self.wire.name

    @name.setter
    def name(self, value: str):
        self.wire.name = value

    @property
    def width(self) -> int:
        return self.wire.width

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

class Cell(Component):
    def __init__(self, cell: Union[rtlil.Cell, "Cell"]):
        if isinstance(cell, Cell):
            self.cell = cell.cell
        elif isinstance(cell, rtlil.Cell):
            self.cell = cell
        else:
            raise ValueError(f"Invalid cell: {cell}")

    @property
    def width(self) -> int:
        return self.cell.parameters.get('Y_WIDTH', None) or self.cell.parameters.get('WIDTH', None)

    def __getitem__(self, index):
        if isinstance(index, int):
            index, _ = _get_slice_params(slice(index, index+1), self.width)
            if index == 0 and self.width == 1:
                return self
        elif isinstance(index, slice):
            start, stop = _get_slice_params(index, self.width)
            if start == 0 and stop == self.width:
                return self
        raise ValueError(f"Cell cannot be sliced with: {index}")

class Slice(Component):
    def __init__(self, wire: Union[rtlil.Wire, Component], start_idx: int = None, stop_idx: int = None):
        offset = 0
        if isinstance(wire, Slice):
            self.wire = wire.wire
            offset = wire.start_idx
        elif isinstance(wire, Component):
            self.wire = wire
        elif isinstance(wire, rtlil.Wire):
            self.wire = Wire(wire)
        else:
            raise ValueError(f"Invalid wire to slice: {wire}")

        start, stop = _get_slice_params(slice(start_idx, stop_idx), wire.width)
        self.start_idx = start + offset
        self.stop_idx = stop + offset

    @property
    def width(self) -> int:
        return self.stop_idx - self.start_idx

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

class Concat(Component):
    def __init__(self, wires: list[Union[rtlil.Wire, Component]]):
        self.parts = self._cleanup(wires)

    @property
    def width(self) -> int:
        return sum(part.width for part in self.parts)

    @classmethod
    def _cleanup(cls, wires) -> list[Component]:
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
                if isinstance(wire, Concat) and wire.parts and isinstance(wire.parts[-1], Const):
                    real_wires.extend(wire.parts[:-1])
                    curr_value = wire.parts[-1].value
                    curr_width = wire.parts[-1].width
                elif isinstance(wire, Concat):
                    real_wires.extend(wire.parts)
                elif isinstance(wire, Component):
                    real_wires.append(wire)
                elif isinstance(wire, rtlil.Wire):
                    real_wires.append(Wire(wire))
                else:
                    raise ValueError(f"Invalid concat part: {wire}")

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

class Const(Component):
    def __init__(self, value: int, width: int):
        self.value = value
        self._width = width

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width: int):
        self._width = width

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

        if isinstance(string, Const):
            return string

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

class MemoryIndex(Component):
    def __init__(self, memory: rtlil.Memory, address: Component):
        if isinstance(memory, rtlil.Memory):
            self.memory = memory
        else:
            raise ValueError(f"Invalid memory: {memory}")

        if isinstance(address, Component):
            self.address = address
        elif isinstance(address, rtlil.Wire):
            self.addrses = Wire(address)
        else:
            raise ValueError(f"Invalid memory address: {address}")

    @property
    def name(self) -> str:
        return self.memory.name

    @property
    def width(self) -> int:
        return self.memory.width

    def __eq__(self, value):
        if isinstance(value, MemoryIndex):
            return (value.memory is self.memory) and (value.address == self.address)
        return False