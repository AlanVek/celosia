 
# celosia

**celosia** is a Python module that provides a way to convert Amaranth HDL designs into usable HDL code. It aims to offer an alternative workflow for generating hardware description language (HDL) files from Amaranth, supporting both Verilog and VHDL output formats.

## What is celosia?

celosia is an independent project designed to help users of [Amaranth HDL](https://amaranth-lang.org/) convert their designs into Verilog and VHDL. This module is not affiliated with Amaranth or Yosys, and is developed as a separate effort. The goal is to provide a simple, Pythonic interface for HDL code generation.

**Disclaimer:** This project is a work in progress and is in not way production-ready. It may lack many features and robustness found in established tools. Use at your own risk.

## Usage

You can find example designs and usage in the [`examples/`](examples/) directory.

Here is a minimal example of how to use celosia to convert an Amaranth design:

```python
from celosia import verilog, vhdl
from amaranth import Module, Signal

def main():
    m = Module()

    a = Signal(8)
    b = Signal(8)
    o = Signal(9)

    m.d.comb += o.eq(a + b)

    ports = [a, b, o]

    # Convert to Verilog
    print('// Verilog')
    verilog_code = verilog.convert(m, ports=ports)
    print(verilog_code)

    # Convert to VHDL
    print('-- VHDL')
    vhdl_code = vhdl.convert(m, ports=ports)
    print(vhdl_code)

if __name__ == '__main__':
    main()
```

For more complete examples, see the files in [`examples/`](examples/).

## Project Status

This project is under active development. Many features are incomplete or experimental. Contributions and feedback are welcome!

## License

See the LICENSE file for details.
