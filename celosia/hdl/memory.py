from amaranth.back import rtlil
from amaranth.hdl import _ast
from celosia.hdl import utils
from typing import Union

class MemoryIndex:
    def __init__(self, name, address, slice: slice = None):
        self.name = name
        self.address = address
        self.slice = slice

class WritePort:
    def __init__(self, cell: rtlil.Cell, ):
        self.cell = cell
        self.name: str = cell.parameters['MEMID']
        self.clk_polarity: bool = cell.parameters['CLK_POLARITY']
        self.portid: int = cell.parameters['PORTID']

        self.width: int = cell.parameters["WIDTH"]

        self.addr: str = None
        self.data: str = None
        self.enable: list[str] = None

    def build(self, signal_map: dict[str, rtlil.Wire], collect_signals):
        self.addr: str = self.cell.ports['ADDR']
        data: str = self.cell.ports['DATA']
        enable = self.cell.ports['EN']
        clk: str = collect_signals(self.cell.ports['CLK'])[0]
        full_en: list[str] = []

        process = rtlil.Process(name = None)

        en_width = en_value = None
        const_params = utils.const_params(enable)
        if const_params is not None:
            en_width, en_value = const_params
            enable = [bool((en_value >> i) & 1) for i in reversed(range(en_width))]
        else:
            enable = collect_signals(enable, raw=True)

        if all(en == enable[0] for en in enable):
            enable = [enable[0]]

        for en in enable[::-1]:
            if isinstance(en, bool):
                full_en.append(en)
            else:
                raw = collect_signals(en, raw=True)
                assert len(raw) == 1, "Internal error"
                raw = raw[0]
                full_en.extend(
                    f'{raw} [{i}]' for i in range(signal_map[raw].width)
                )

        self.enable = [
            f"1'{int(en)}" if isinstance(en, bool) else en for en in full_en
        ]

        if self.width % len(full_en):
            raise RuntimeError("Invalid enables for write port of memory")

        const_params = utils.const_params(data)
        data_value = None
        allow_slice = False
        if const_params is not None:
            _, data_value = const_params
            self.data = data
        else:
            new_data = collect_signals(data)
            if len(new_data) == 1:
                allow_slice = True
                self.data = new_data[0]     # TODO: Maybe check that it's not a slice already? Otherwise it will fail
            else:
                assert len(full_en) <= 1, "Cannot take slice of concatenation"
                self.data = data

        chunk_width = self.width // len(full_en)
        start_idx = 0
        for enable in full_en:
            if len(full_en) == 1:
                part = None
            else:
                part = slice(start_idx, start_idx + chunk_width)
            lhs = MemoryIndex(self.name, self.addr, part)

            if data_value is None:
                if allow_slice:
                    rhs = f'{self.data} [{start_idx+chunk_width-1}:{start_idx}]'
                else:
                    rhs = f'{self.data}'
            else:
                rhs = _ast.Const((data_value >> start_idx) & int('1' * chunk_width, 2), chunk_width)

            if enable == True:
                process.assign(lhs, rhs)
            elif enable != False:
                process.switch(enable).case(['1']).assign(lhs, rhs)

            start_idx += chunk_width

        return (clk, process)

class ReadPort:
    def __init__(self, cell: rtlil.Cell, portid: int):
        self.cell = cell

        self.portid = portid
        self.name: str = cell.parameters['MEMID']
        self.width: int = cell.parameters["WIDTH"]
        self.clk_enable: bool = cell.parameters['CLK_ENABLE']
        self.clk_polarity: bool = cell.parameters['CLK_POLARITY']
        self.transparency_mask: int = cell.parameters['TRANSPARENCY_MASK'].value

        self.proxy: rtlil.Wire = None

    def build(self, collect_signals, new_signal_creator, write_ports: dict[int, WritePort] = None):
        write_ports = write_ports or {}
        ret_process: tuple[str, rtlil.Process] = None
        connection: tuple[str, Union[str, MemoryIndex]] = None

        enable: str = self.cell.ports['EN']
        data: str = self.cell.ports['DATA']
        addr: str = self.cell.ports['ADDR']

        if self.clk_enable:
            clk = collect_signals(self.cell.ports['CLK'])[0]
            process = rtlil.Process(name=None)

            self.proxy = new_signal_creator(self.width, name = f'_{self.portid}_')

            const_params = utils.const_params(enable)
            if const_params is None:
                assigner = process.switch(enable).case('1')
            else:
                width, value = const_params
                if value & 1:
                    assigner = process
                else:
                    return None, (data, f"{self.width}'0")

            assigner.assign(self.proxy.name,  MemoryIndex(self.name, addr))

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

                chunk_width = self.width // len(wp.enable)
                start_idx = 0
                for en in wp.enable:

                    if wp.addr is None or wp.data is None:
                        start_idx += chunk_width
                        continue

                    if len(wp.enable) == 1:
                        index = ''
                    else:
                        index = f' [{start_idx+chunk_width-1}:{start_idx}]'

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

                    const_params = utils.const_params(wp.data)
                    if const_params is not None:
                        _, data_value = const_params
                        wp_data = (data_value >> start_idx) & int('1' * chunk_width, 2)
                        wp_data = f"{chunk_width}'{wp_data}"
                    else:
                        wp_data = f'{wp.data}{index}'

                    assigner.switch(sel).case(['1']).assign(f'{self.proxy.name}{index}', f'{wp_data}')

                    start_idx += chunk_width

            ret_process = (clk, process)
            connection = (data, self.proxy.name)

        else:
            connection = (data, MemoryIndex(self.name, addr))

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
        init: str = cell.ports.get('DATA', None)
        if init is not None:
            mem_size = self.memory.depth * self.width

            header = f"{mem_size}'"
            if not init.startswith(header):
                raise RuntimeError(f"Invalid memory initializer: {init}")

            bits = init[len(header):]
            if len(bits) != mem_size:
                raise RuntimeError(f"Invalid memory initializer: {len(bits)} != {mem_size}")

            init_data: list[_ast.Const] = []
            for i in range(0, len(bits), self.width):
                init_data.append(_ast.Const(int(bits[i : i + self.width], 2), self.width))
            self.attributes['init'] = init_data[::-1]

    def add_wp(self, wp: rtlil.Cell):
        self.wps.append(WritePort(wp))

    def add_rp(self, rp: rtlil.Cell):
        self.rps.append(ReadPort(rp, len(self.rps)))

    def build(self, signal_map: dict[str, rtlil.Wire], collect_signals, new_signal_creator):
        processes: list[tuple[rtlil.Process, str]] = []
        connections: list[tuple[str, str]] = []

        write_ports = {}

        for wp in self.wps:
            new_process = wp.build(signal_map, collect_signals)
            if new_process is not None:
                processes.append(new_process)
            write_ports[wp.portid] = wp

        for rp in self.rps:
            new_process, new_connection = rp.build(collect_signals, new_signal_creator, write_ports)

            if new_process is not None:
                processes.append(new_process)
            connections.append(new_connection)

        return processes, connections

