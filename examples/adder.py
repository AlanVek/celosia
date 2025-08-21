from amaranth import *

class Adder(Elaboratable):
    def __init__(self, width, domain='sync'):   # Use comb or None for combinational
        if width <= 0:
            raise RuntimeError(f"Invalid width {width}, must be > 0")

        if domain is None:
            domain = 'comb'

        self.width  = width
        self.domain = domain

        self.a      = Signal(width)
        self.b      = Signal(width)
        self.o      = Signal(width + 1)

    def elaborate(self, platform):
        m = Module()

        m.d[self.domain] += self.o.eq(self.a + self.b)

        return m

def main():
    from celosia import verilog, vhdl
    import os

    path = os.path.dirname(__file__)

    adder = Adder(4, domain = 'sync')
    ports = [
        adder.a, adder.b, adder.o,
    ]

    with open(os.path.join(path, 'adder.v'), 'w') as f:
        f.write(verilog.convert(adder, ports = ports))

    with open(os.path.join(path, 'adder.vhd'), 'w') as f:
        f.write(vhdl.convert(adder, ports = ports))

if __name__ == '__main__':
    main()