import os
import importlib.util
from celosia import verilog, vhdl
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=False, default=None, help='Base path to find tests (defaults to .)')
    parser.add_argument('-o', '--output', required=True, default=None, help='Output directory to store results')
    parser.add_argument('--verilog', required=False, action='store_true', default=False, help='Convert to verilog')
    parser.add_argument('--vhdl', required=False, action='store_true', default=False, help='Convert to VHDL')
    parser.add_argument('--all', required=False, action='store_true', default=False, help='Convert to all HDL languages')

    args = parser.parse_args()

    test_path = args.path
    if test_path is None:
        test_path = Path(__file__).parent

    test_path = Path(test_path)
    output_path = Path(args.output)

    os.makedirs(str(output_path), exist_ok=True)

    if not test_path.is_dir():
        raise RuntimeError(f"Unknown test path: {test_path}")

    hdl_mappings = [
        (verilog.convert, '.v', args.verilog),
        (vhdl.convert, '.vhd', args.vhdl),
    ]

    namespace = {}
    for file in test_path.rglob('*.py'):

        # Exclude self
        if file.resolve() == Path(__file__).resolve():
            continue

        namespace.clear()

        try:
            module_name = file.stem
            spec = importlib.util.spec_from_file_location(module_name, str(file))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            test = getattr(module, 'test', None)
            if not callable(test):
                print(f"Skipping {file}, test not defined")
                continue

            output_template = output_path / file.relative_to(test_path)
            os.makedirs(output_template.parent, exist_ok=True)
            module, ports = test()

            for function, extension, condition in hdl_mappings:
                if condition or args.all:
                    output_template.with_suffix(extension).write_text(function(module, ports=ports))

        except BaseException as e:
            print(f"Skipping {file}, test failed: {e}")

if __name__ == '__main__':
    main()
