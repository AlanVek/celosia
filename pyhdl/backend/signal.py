import pyhdl.backend.statement as pyhdl_statement
from amaranth.hdl import ast, ir
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyhdl.hdl import HDL

class Signal:
    def __init__(self, signal: ast.Signal, domain: ir.ClockDomain = None):
        self.signal = signal
        self.domain = domain
        self.statements = []

        self._no_reset_statement = False

    def disable_reset_statement(self):
        self._no_reset_statement = True

    @property
    def assigned_bits(self):
        assigned_bits = [False] * len(self.signal)
        for statement in self.statements:
            if isinstance(statement, pyhdl_statement.Assign):
                start_idx = statement._start_idx
                stop_idx = statement._stop_idx

                if start_idx is None:
                    start_idx = 0
                if stop_idx is None:
                    stop_idx = len(self.signal)

                for bit in range(start_idx, stop_idx):
                    if assigned_bits[bit]:
                        pass    # TODO: Possible Verilog error?
                    assigned_bits[bit] = True

        return assigned_bits

    @property
    def reset_statement(self):
        if self.domain is not None or self._no_reset_statement:
            return None

        assigned_bits = self.assigned_bits
        if all(assigned_bits):
            return None

        return pyhdl_statement.Assign(ast.Const(self.signal.reset, len(self.signal)))

    @staticmethod
    def sanitize(name, hdl: "HDL" = None):
        if hdl is not None:
            name = hdl.sanitize(name)
        return name

    def add_statement(self, statement):
        if not isinstance(statement, list):
            statement = [statement]
        self.statements.extend(statement)

    @property
    def static(self):
        # TODO: This can be used to treat comb-only signals without "if" as assigns

        if self.domain is None:
            if self.statements:
                if len(self.statements) <= 1 and all(isinstance(st, pyhdl_statement.Assign) for st in self.statements):
                    if all(self.assigned_bits):
                        return True
            else:
                return True

        return False

    def shape(self):
        return self.signal.shape()

    def _rhs_signals(self):
        return [self.signal]

class RemappedSignal(Signal):
    def __init__(self, signal: ast.Signal, sync_signal: ast.Signal):
        super().__init__(signal, domain=None)
        self.sync_signal = sync_signal

    @property
    def reset_statement(self) -> pyhdl_statement.Assign:
        reset = super().reset_statement

        if reset is not None:
            reset.rhs = self.sync_signal

        return reset

class Port(Signal):
    def __init__(self, signal: ast.Signal, direction: str, domain: ir.ClockDomain = None):
        super().__init__(signal, domain)
        self.direction = direction

        # if self.direction == 'i':
        #     self.reset_statement = None

class Memory(Signal):
    def __init__(self, signal: ast.Signal, init: list = None):
        super().__init__(signal)
        self.init = [] if init is None else init

    @property
    def reset_statement(self):
        return None

class MemoryPort(Signal):
    def __init__(self, signal: ast.Signal, index: ast.Signal, domain: ir.ClockDomain = None):
        super().__init__(signal, domain)
        self.index = index
        self.domain = domain

    @property
    def reset_statement(self):
        return None