import celosia.backend.statement as celosia_statement
from amaranth.hdl import ast, ir

class Signal:
    def __init__(self, signal: ast.Signal, domain: ir.ClockDomain = None):
        self.signal = signal
        self.domain = domain
        self.statements: list[celosia_statement.Statement] = []

        self._no_reset_statement = False

    def disable_reset_statement(self):
        self._no_reset_statement = True

    @property
    def assigned_bits(self):
        assigned_bits = [False] * len(self.signal)
        for statement in self.statements:
            if isinstance(statement, celosia_statement.Assign):
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

        return celosia_statement.Assign(ast.Const(self.signal.reset, len(self.signal)))

    def add_statement(self, statement):
        if not isinstance(statement, list):
            statement = [statement]
        self.statements.extend(statement)

    @property
    def static(self):
        # TODO: This can be used to treat comb-only signals without "if" as assigns

        if self.domain is None:
            if self.statements:
                if len(self.statements) <= 1 and all(isinstance(st, celosia_statement.Assign) for st in self.statements):
                    if all(self.assigned_bits):
                        return True
            else:
                return True

        return False

    def shape(self):
        return self.signal.shape()

    def _rhs_signals(self):
        return self.signal._rhs_signals()

    def __len__(self):
        return len(self.signal)

    @property
    def name(self) -> str:
        return self.signal.name

    @name.setter
    def name(self, value:str) -> None:
        self.signal.name = value

    @property
    def attrs(self) -> dict:
        return self.signal.attrs

class RemappedSignal(Signal):
    def __init__(self, signal: ast.Signal, sync_signal: ast.Signal):
        super().__init__(signal, domain=None)
        self.sync_signal = sync_signal

    @property
    def reset_statement(self) -> celosia_statement.Assign:
        reset = super().reset_statement

        if reset is not None:
            reset.rhs = self.sync_signal

        return reset

class Port(Signal):
    def __init__(self, signal: ast.Signal, direction: str, domain: ir.ClockDomain = None):
        super().__init__(signal, domain)
        self.direction = direction
        self._alt_name: str = None

        # if self.direction == 'i':
        #     self.reset_statement = None

    def set_alt_name(self, value: str):
        self._alt_name = value

    @property
    def name(self) -> str:
        if self._alt_name is not None:
            return self._alt_name
        return super().name

    @name.setter
    def name(self, value:str) -> None:
        self.signal.name = value

class Memory(Signal):
    def __init__(self, signal: ast.Signal, init: list = None, attrs: dict = None):
        super().__init__(signal)
        self.init = [] if init is None else init
        self._attrs = {} if attrs is None else attrs

    @property
    def reset_statement(self):
        return None

    @property
    def attrs(self) -> dict:
        return self._attrs

class MemoryPort(Signal):
    def __init__(self, signal: ast.Signal, memory: Memory, index: ast.Signal, domain: ir.ClockDomain = None):
        super().__init__(signal, domain)
        self.memory = memory
        self.index = index
        self.domain = domain

    @property
    def reset_statement(self):
        return None
