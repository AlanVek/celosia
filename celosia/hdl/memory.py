from amaranth.back import rtlil

class MemoryIndex:
    def __init__(self, name, address, slice: slice = None):
        self.name = name
        self.address = address
        self.slice = slice

class WritePort:
    def __init__(self, cell: rtlil.Cell, signal_map: dict[str, rtlil.Wire], collect_signals):
        self.cell = cell
        self.name: str = cell.parameters['MEMID']
        self.clk_polarity: bool = cell.parameters['CLK_POLARITY']
        self.portid: int = cell.parameters['PORTID']

        self.width: int = cell.parameters["WIDTH"]

        self.signal_map = signal_map
        self.collect_signals = collect_signals

        self.addr: str = None
        self.data: str = None
        self.enable: list[str] = None
        self.clk: str  = None

    def build(self):
        self.addr = self.collect_signals(self.cell.ports['ADDR'], raw=True)[0]
        self.data = self.collect_signals(self.cell.ports['DATA'], raw=True)[0]
        enable = self.collect_signals(self.cell.ports['EN'], raw=False)
        self.clk = self.collect_signals(self.cell.ports['CLK'], raw=True)[0]
        full_en: list[str] = []

        if self.width % len(enable):
            raise RuntimeError("Invalid enables for write for of memory")

        process = rtlil.Process(name = None)

        if all(en == enable[0] for en in enable):
            enable = [enable[0]]

        for en in enable[::-1]:
            raw = self.collect_signals(en, raw=True)[0]
            full_en.extend(
                f'{raw} [{i}]' for i in range(self.signal_map[raw].width)
            )

        self.enable = full_en

        chunk_width = self.width // len(full_en)
        start_idx = 0
        for enable in full_en:
            if len(full_en) == 1:
                part = None
            else:
                part = slice(start_idx, start_idx + chunk_width)

            process.switch(enable).case(['1']).assign(MemoryIndex(self.name, self.addr, part), f'{self.data} [{start_idx+chunk_width-1}:{start_idx}]')
            start_idx += chunk_width

        return [(self.clk, process)], []

class ReadPort:
    def __init__(self, cell: rtlil.Cell, portid: int, collect_signals):
        self.cell = cell

        self.portid = portid
        self.name: str = cell.parameters['MEMID']
        self.width: int = cell.parameters["WIDTH"]
        self.clk_enable: bool = cell.parameters['CLK_ENABLE']
        self.clk_polarity: bool = cell.parameters['CLK_POLARITY']
        self.transparency_mask: int = cell.parameters['TRANSPARENCY_MASK'].value

        self.clk: str = None

        self.collect_signals = collect_signals

    def build(self, new_signal_creator, write_ports: dict[int, WritePort] = None):
        write_ports = write_ports or {}

        addr: str = self.collect_signals(self.cell.ports['ADDR'], raw=True)[0]
        data: str = self.collect_signals(self.cell.ports['DATA'], raw=True)[0]
        enable: list[str] = self.collect_signals(self.cell.ports['EN'], raw=True)

        processes: list[rtlil.Process] = []
        connections: list[tuple[str, str]] = []
        new_signals: list[rtlil.Wire] = []

        if self.clk_enable:
            self.clk = self.collect_signals(self.cell.ports['CLK'], raw=True)[0]
            process = rtlil.Process(name=None)

            # TODO: Validate that name is not in use?
            proxy: rtlil.Wire = new_signal_creator(self.width, name = f'_{self.portid}_')
            new_signals.append(proxy)

            if enable:
                assigner = process.switch(enable[0]).case('1')

            else:  # Const, always enabled
                assigner = process

            assigner.assign(proxy.name,  MemoryIndex(self.name, addr))

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

                chunk_width = self.width // len(wp.enable)
                start_idx = 0
                for en in wp.enable:
                    if len(wp.enable) == 1:
                        part = None
                    else:
                        part = slice(start_idx, start_idx + chunk_width)

                    sel = f'{en} && ({wp.addr} == {addr})'  # WIP: AND and CMP must be generic

                    assigner.switch(sel).case(['1']).assign(f'{proxy.name} [{start_idx+chunk_width-1}:{start_idx}]', MemoryIndex(self.name, addr, part))
                    start_idx += chunk_width

                port_id += 1

            processes.append((self.clk, process))
            connections.append((data, proxy.name))

        else:
            connections.append((data, MemoryIndex(self.name, addr)))

        return processes, connections, new_signals

class Memory:
    def __init__(self, name: str, memory: rtlil.Memory, signal_map: dict[str, rtlil.Wire], collect_signals):
        self.name = name
        self.memory = memory
        self.cell: rtlil.Cell = None
        self.rps: list[ReadPort] = []
        self.wps: list[WritePort] = []

        self.signal_map = signal_map
        self.collect_signals = collect_signals

    def set_cell(self, cell: rtlil.Cell):
        if self.cell is not None:
            raise RuntimeError(f"Trying to reset Memory cell for {self.memory.name}")
        self.cell = cell

    def add_wp(self, wp: rtlil.Cell):
        self.wps.append(WritePort(wp, self.signal_map, self.collect_signals))

    def add_rp(self, rp: rtlil.Cell):
        self.rps.append(ReadPort(rp, len(self.rps), self.collect_signals))

    @property
    def depth(self) -> int:
        return self.cell.parameters.get('WORDS', None)

    @property
    def width(self) -> int:
        return self.cell.parameters.get('WIDTH', None)

    def build(self, new_signal_creator):
        processes: list[rtlil.Process] = []
        connections: list[tuple[str, str]] = []
        signals: list[rtlil.Wire] = []

        write_ports = {}

        for wp in self.wps:
            new_processes, new_connections = wp.build()
            processes.extend(new_processes)
            connections.extend(new_connections)

            write_ports[wp.portid] = wp

        for rp in self.rps:
            new_processes, new_connections, new_signals = rp.build(new_signal_creator, write_ports)
            processes.extend(new_processes)
            connections.extend(new_connections)
            signals.extend(new_signals)

        return processes, connections, signals

