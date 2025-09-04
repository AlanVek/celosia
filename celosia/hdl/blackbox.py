from typing import Any, Union
from amaranth.hdl import ast

class BlackBoxEntry:
    valid_types = ()

    def __init__(self, type: str):
        self.validate_type(type)
        self.type: str = type

    @classmethod
    def validate_type(cls, type: str):
        if type not in cls.valid_types:
            raise ValueError(f"Unknown parameter type: {type}. Expected one of: {', '.join(cls.valid_types)}")

    def convert(self, value: Any) -> Any:
        return value

class BlackBoxParameter(BlackBoxEntry):
    valid_types = ('integer', 'string', 'natural', 'positive', 'boolean', 'signed', 'unsigned', 'real')
    default_type = 'integer'

    def _value_error(self, value: Any, checks: Union[type, tuple[type, ...]] = None):
        fail = True
        if checks is not None:
            fail = not isinstance(value, checks)

        if fail:
            raise ValueError(f"Invalid value for {self.type} parameter: {value}")

    def convert(self, value: Any) -> Any:
        if self.type == 'string':
            self._value_error(value, str)
        elif self.type in ('integer', 'natural', 'positive', 'signed', 'unsigned'):
            self._value_error(value, (int, ast.Const))
        elif self.type == 'real':
            self._value_error(value, (int, float, ast.Const))
        elif self.type == 'boolean':
            self._value_error(value, (int, bool, ast.Const))

        return value

class BlackBoxPort(BlackBoxEntry):
    valid_types = ('std_logic', 'std_logic_vector', 'signed', 'unsigned')
    default_type = 'std_logic_vector'

    def __init__(self, type: str, direction: str, width: int = None):
        super().__init__(type)

        if direction not in ('i', 'o', 'io'):
            raise ValueError(f"Invalid port direction: {direction}")

        self.direction = direction
        self.width = self.validate_width(type, width)

    @classmethod
    def validate_width(cls, type: str, width: int) -> int:
        if type in ('std_logic',):
            if not (width is None or width == 1):
                raise RuntimeError(f"Invalid width for {type}: {width}")
            width = 1
        elif width is None:
            raise RuntimeError(f"Missing width for {type}")

        return width

    def convert(self, value: Any) -> Any:
        if self.type == 'std_logic':
            if isinstance(value, ast.Const):
                pass
            elif isinstance(value, ast.Signal):
                if len(value) == 1:
                    value = value[0]
            else:
                raise ValueError(f"Invalid value for {self.type} port: {value}")

        elif isinstance(value, ast.Value):
            signed = value.shape().signed
            if self.type == 'unsigned' and signed:
                value = value.as_unsigned()
            elif self.type == 'signed' and not signed:
                value = value.as_signed()

        return value

class BlackBox:
    def __init__(self, description: dict[str, Union[int, str, tuple[str, int]]]):
        self.parameters: dict[str, BlackBoxParameter] = {}
        self.ports: dict[str, BlackBoxPort] = {}

        for key, value in description.items():
            if isinstance(value, tuple):
                if len(value) != 2:
                    raise ValueError(f"Invalid blackbox entry: {value}. Expected tuple of length 2")
                type, width = value
            elif isinstance(value, str):
                type = value
                width = None
            elif isinstance(value, int):
                type = None
                width = value
            else:
                raise ValueError(f"Invalid blackbox entry: {value}")

            if key.startswith(('i_', 'o_', 'io_')):
                port_dir, port_name = key.split('_', maxsplit=1)
                if type is None:
                    type = BlackBoxPort.default_type
                self.ports[port_name] = BlackBoxPort(type, port_dir, width)

            elif key.startswith('p_'):
                param_name = key.split('_', maxsplit=1)[1]
                self.parameters[param_name] = BlackBoxParameter(type)

            elif key.startswith('a_'):
                # TODO: Attributes
                pass

            else:
                raise ValueError(f"Invalid blackbox description entry: {key}")
    
