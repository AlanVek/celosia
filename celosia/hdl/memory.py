from amaranth.back import rtlil
from amaranth.hdl import _ast
import celosia.hdl.wire as celosia_wire
from typing import Union

class WritePort:
    def __init__(self, cell: rtlil.Cell, memory: rtlil.Memory):
        self.cell = cell
        self.memory = memory

        self.name: str = cell.parameters['MEMID']
        self.clk_polarity: bool = cell.parameters['CLK_POLARITY']
        self.portid: int = cell.parameters['PORTID']

        self.width: int = cell.parameters["WIDTH"]

        self.addr: str = None
        self.data: str = None
        self.enable: list[str] = None

        assert self.name == memory.name

    def build(self):
        self.addr = self.cell.ports['ADDR']
        self.data = self.cell.ports['DATA']
        enable = self.cell.ports['EN']
        clk = self.cell.ports['CLK']

        process = rtlil.Process(name = None)

        if all(enable[i] == enable[0] for i in range(enable.width)):
            enable = enable[0]

        full_en = [enable[i] for i in range(enable.width)]
        self.enable = enable

        if self.width % len(full_en):
            raise RuntimeError("Invalid enables for write port of memory")

        chunk_width = self.width // len(full_en)
        start_idx = 0
        for enable in full_en:
            if len(full_en) == 1:
                part = None
            else:
                part = slice(start_idx, start_idx + chunk_width)

            lhs = celosia_wire.MemoryIndex(self.memory, self.addr)
            if part is not None:
                lhs = celosia_wire.Slice(lhs, part.start, part.stop)
            rhs = self.data[start_idx : start_idx + chunk_width]
            if enable == True:
                process.assign(lhs, rhs)
            elif enable != False:
                process.switch(enable).case(['1']).assign(lhs, rhs)

            start_idx += chunk_width

        return (clk, process)

class ReadPort:
    def __init__(self, cell: rtlil.Cell, memory: rtlil.Memory, portid: int):
        self.cell = cell
        self.memory = memory

        self.portid = portid
        self.name: str = cell.parameters['MEMID']
        self.width: int = cell.parameters["WIDTH"]
        self.clk_enable: bool = cell.parameters['CLK_ENABLE']
        self.clk_polarity: bool = cell.parameters['CLK_POLARITY']
        self.transparency_mask: int = cell.parameters['TRANSPARENCY_MASK'].value

        self.proxy: celosia_wire.Wire = None

        assert self.name == memory.name

    def build(self, new_signal_creator, write_ports: dict[int, WritePort] = None):
        write_ports = write_ports or {}
        ret_process: tuple[str, rtlil.Process] = None
        connection: tuple[str, Union[str, celosia_wire.MemoryIndex]] = None

        enable = self.cell.ports['EN']
        data = self.cell.ports['DATA']
        addr = self.cell.ports['ADDR']

        if self.clk_enable:
            clk = self.cell.ports['CLK']
            process = rtlil.Process(name=None)

            self.proxy = celosia_wire.Wire(new_signal_creator(self.width, name = f'_{self.portid}_'))

            if not isinstance(enable, celosia_wire.Const):
                assigner = process.switch(enable).case('1')
            elif enable[0].value:
                assigner = process
            else:
                return None, (data, celosia_wire.Const(0, self.width))

            assigner.assign(self.proxy.name, celosia_wire.MemoryIndex(self.memory, addr))

            port_id = 0
            while True:
                shifted = self.transparency_mask >> port_id

                if not shifted:
                    break

                if (shifted & 1):
                    pass

                if port_id not in write_ports:
                    raise RuntimeError("Read port received invalid transparency mask!")

                wp = write_ports[port_id]
                port_id += 1

                if wp.enable is None:
                    continue

                chunk_width = self.width // wp.enable.width
                start_idx = 0

                for i in range(wp.enable.width):
                    en = wp.enable[i]

                    if wp.addr is None or wp.data is None:
                        start_idx += chunk_width
                        continue

                    # en && (wp.addr == addr)
                    sel = rtlil.Cell(
                        '$and',
                        name = None,
                        ports = {
                            'A': en,
                            'B': rtlil.Cell(
                                '$eq',
                                name = None,
                                ports = {
                                    'A': wp.addr,
                                    'B': addr,
                                },
                                parameters = {
                                    'A_SIGNED': False,
                                    'A_WIDTH': wp.cell.parameters['ABITS'],
                                    'B_SIGNED': False,
                                    'B_WIDTH': self.cell.parameters['ABITS'],
                                    'Y_WIDTH': 1,
                                }
                            ),
                        },
                        parameters = {
                            'A_SIGNED': False,
                            'A_WIDTH': 1,
                            'B_SIGNED': False,
                            'B_WIDTH': 1,
                            'Y_WIDTH': 1,
                        },
                    )

                    assigner.switch(sel).case(['1']).assign(self.proxy[start_idx : start_idx + chunk_width], wp.data[start_idx : start_idx + chunk_width])

                    start_idx += chunk_width

            ret_process = (clk, process)
            connection = (data, self.proxy.name)

        else:
            connection = (data, celosia_wire.MemoryIndex(self.memory, addr))

        return ret_process, connection

class Memory(rtlil.Wire):
    def __init__(self, memory: rtlil.Memory):
        attributes = memory.attributes.copy()

        attributes['init'] = [_ast.Const(0, memory.width) for _ in range(memory.depth)]

        super().__init__(memory.width, name = memory.name, attrs=attributes)

        self.memory = memory
        self.rps: list[ReadPort] = []
        self.wps: list[WritePort] = []

    def set_cell(self, cell: rtlil.Cell):
        init: celosia_wire.Const = celosia_wire.Const.from_string(cell.ports.get('DATA', None))
        if init is not None:
            mem_size = self.memory.depth * self.width

            if init.width != mem_size:
                raise RuntimeError(f"Invalid memory initializer: {init.width} != {mem_size}")

            self.attributes['init'] = [
                init[i : i + self.width] for i in range(0, mem_size, self.width)
            ]

    def add_wp(self, wp: rtlil.Cell):
        self.wps.append(WritePort(wp, self.memory))
        return self.wps[-1]

    def add_rp(self, rp: rtlil.Cell, id: int):
        self.rps.append(ReadPort(rp, self.memory, id))
        return self.rps[-1]

    def build(self, new_signal_creator):
        processes: list[tuple[str, rtlil.Process]] = []
        connections: list[tuple[str, str]] = []

        write_ports = {}

        for wp in self.wps:
            new_process = wp.build()
            if new_process is not None:
                processes.append(new_process)
            write_ports[wp.portid] = wp

        for rp in self.rps:
            new_process, new_connection = rp.build(new_signal_creator, write_ports)

            if new_process is not None:
                processes.append(new_process)
            connections.append(new_connection)

        return processes, connections

