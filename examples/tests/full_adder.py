from amaranth import *

class SingleAdder(Elaboratable):
    def __init__(self):
        self.a = Signal()
        self.b = Signal()
        self.ci = Signal()

        self.o = Signal()
        self.co = Signal()

    def elaborate(self, platform):
        m = Module()

        m.d.comb += [
            self.o.eq(self.a ^ self.b ^ self.ci),
            self.co.eq(
                (self.a & self.b) | (self.a & self.ci) | (self.b & self.ci),
            )
        ]

        return m

class FullAdder(Elaboratable):
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

        adders = [SingleAdder() for _ in range(self.width)]

        ci  = co    = Const(0, 1)
        o           = []

        for i, adder in enumerate(adders):
            m.submodules[f'adder{i}'] = adder

            m.d.comb += [
                adder.ci    .eq(ci),
                adder.a     .eq(self.a[i]),
                adder.b     .eq(self.b[i]),
            ]

            o.append(adder.o)
            co = adder.co

        m.d[self.domain] += self.o.eq(Cat(*o, co))

        return m

def test():
    adder = FullAdder(4, domain = 'sync')
    ports = [
        adder.a, adder.b, adder.o,
    ]
    return adder, ports