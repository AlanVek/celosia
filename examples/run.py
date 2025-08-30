import os
import importlib.util
import celosia
from pathlib import Path
import argparse
import logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=False, default=None, help='Base path to find tests. Can be a file or a directory (defaults to .)')
    parser.add_argument('-r', '--recursive', required=False, action='store_true', default=False, help='If input is a directory, look for test files recursively')
    parser.add_argument('-o', '--output', required=True, default=None, help='Output directory to store results')
    parser.add_argument('--all', required=False, action='store_true', default=False, help='Convert to all HDL languages')

    lang_map = celosia.get_lang_map()
    for name in lang_map.keys():
        parser.add_argument(f'--{name}', required=False, action='store_true', default=False, help=f'Convert to {name}')

    args = parser.parse_args()

    test_path = args.input
    if test_path is None:
        test_path = Path(__file__).parent
    test_path = Path(test_path)

    if test_path.is_dir():
        input_files = (test_path.rglob if args.recursive else test_path.glob)('*.py')
    elif test_path.is_file():
        input_files = [test_path]
        test_path = test_path.parent
    else:
        raise ValueError(f"Invalid input path: {test_path}")

    output_path = Path(args.output)

    os.makedirs(str(output_path), exist_ok=True)

    hdl_mappings = []
    for name, HDLType in lang_map.items():
        hdl_mappings.append(
            (HDLType().convert, f'.{HDLType.default_extension}', getattr(args, name))
        )

    for file in input_files:

        # Exclude self
        if file.resolve() == Path(__file__).resolve():
            continue

        try:
            module_name = file.stem
            spec = importlib.util.spec_from_file_location(module_name, str(file))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            test = getattr(module, 'test', None)
            if not callable(test):
                logging.warning(f"Skipping {file}, test not defined")
                continue

            output_template = output_path / file.relative_to(test_path)
            os.makedirs(output_template.parent, exist_ok=True)
            module, ports = test()

            for function, extension, condition in hdl_mappings:
                if condition or args.all:
                    output_template.with_suffix(extension).write_text(function(module, ports=ports))

        except BaseException as e:
            logging.warning(f"Skipping {file}, test failed: {e}")

if __name__ == '__main__':
    main()
