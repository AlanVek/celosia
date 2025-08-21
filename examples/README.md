# Celosia Examples

This directory contains example scripts and tests for getting started with the `celosia` package.

## Quick Start

The main entry point is `run.py`, which automates HDL generation from Amaranth modules defined in test scripts.

### What does `run.py` do?
- Searches for all `.py` files in the specified directory (defaults to its own directory if none is provided).
- For each file, looks for a `test` function.
- If found, runs the `test` function, which should return a tuple: `[Amaranth module, Amaranth ports]`.
- Generates HDL (Verilog, VHDL, or both) for the module and ports.
- Output files are saved in the specified output directory, preserving the test file hierarchy.

### Usage

```bash
python run.py [--verilog] [--vhdl] [--all] [--output <output_dir>] [--input <file_or_dir>] [-r|--recursive]
```

- `--verilog`: Generate Verilog files
- `--vhdl`: Generate VHDL files
- `--all`: Selects all supported HDL languages (Verilog, VHDL, and any future additions) so you don't have to specify each one individually.
- `--output <output_dir>`: Directory to store generated files
- `--input <file_or_dir>`: Path to a test file or directory containing test scripts (defaults to the same directory as `run.py` if not provided)
- `-r`, `--recursive`: If `--input` is a directory, search for test scripts recursively in all subdirectories. If not specified, only the top-level directory is searched.

### Writing Your Own Test
1. Create a `.py` file in your test directory.
2. Define a `test` function that returns `[Amaranth module, Amaranth ports]`.
   ```python
   from amaranth import Module, Signal
   def test():
       m = Module()
       a = Signal()
       b = Signal()
       # ... define your module ...
       return m, [a, b]
   ```
3. Run `run.py` to generate HDL for your module.

---
For more details, see the main project README or explore the example test scripts in `examples/tests/`.
