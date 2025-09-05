from amaranth.back import rtlil
from typing import Union

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

        contents: list[rtlil.Process] = []
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

            process.switch(enable).case(['1']).assign(MemoryIndex(self.name, self.addr, part), self.data)
            contents.append(process)
            start_idx += chunk_width

        return contents

class ReadPort:
    def __init__(self, cell: rtlil.Cell, write_ports: dict[int, WritePort] = None):
        self.write_ports = write_ports or {}

        self.addr: str = cell.ports['ADDR']
        self.data: str = cell.ports['DATA']
        self.en: str = cell.ports['EN']
        self.clk: str = cell.ports['CLK']

        self.clk_enable: bool = cell.parameters['CLK_ENABLE']
        self.clk_polarity: bool = cell.parameters['CLK_POLARITY']

        # if isinstance(cell, _nir.AsyncReadPort):
        #     transparency_mask = 0
        # if isinstance(cell, _nir.SyncReadPort):
        #     transparency_mask = sum(
        #         1 << memory_info.write_port_ids[write_port_cell_index]
        #         for write_port_cell_index in cell.transparent_for
        #     )
        # parameters = {
        #     "MEMID": memory_info.memid,
        #     "ABITS": len(cell.addr),
        #     "WIDTH": cell.width,
        #     "TRANSPARENCY_MASK": _ast.Const(transparency_mask, memory_info.num_write_ports),
        # }
        # if isinstance(cell, _nir.AsyncReadPort):
        #     ports.update({
        #         "EN": self.sigspec(_nir.Net.from_const(1)),
        #         "CLK": self.sigspec(_nir.Net.from_const(0)),
        #     })
        #     parameters.update({
        #         "CLK_ENABLE": False,
        #         "CLK_POLARITY": True,
        #     })
        # if isinstance(cell, _nir.SyncReadPort):
        #     ports.update({
        #         "EN": self.sigspec(cell.en),
        #         "CLK": self.sigspec(cell.clk),
        #     })
        #     parameters.update({
        #         "CLK_ENABLE": True,
        #         "CLK_POLARITY": {
        #             "pos": True,
        #             "neg": False,
        #         }[cell.clk_edge],
            # })

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
        self.rps.append(ReadPort(rp))

    @property
    def depth(self) -> int:
        return self.cell.parameters.get('WORDS', None)

    @property
    def width(self) -> int:
        return self.cell.parameters.get('WIDTH', None)

    def build(self):
        for wp in self.wps:
            wp.build()


