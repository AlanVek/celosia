from amaranth.hdl import ast

class Statement:
    pass

class Assign(Statement):
    def __init__(self, rhs: ast.Value, start_idx: int = None, stop_idx: int = None):
        self.rhs = rhs
        self._start_idx = start_idx
        self._stop_idx = stop_idx

class Switch(Statement):

    class Case:
        def __init__(self, test: ast.Value | None, statements: list[Statement]):
            self.test = test
            self.statements = statements

    class If(Case):
        pass

    class Else(Case):
        def __init__(self, statements: list[Statement]):
            super().__init__(None, statements)

    def __init__(self, test: ast.Value, cases: dict[int | str | tuple | None, list[Statement]]):
        self.test = test
        self.cases = self.process_cases(test, cases)

        # Move default to last place
        default = self.cases.pop(None, None)
        if default is not None:
            self.cases[None] = default

        self.strip_unused_cases()

        self.as_if = self._as_if()

    def strip_unused_cases(self) -> None:
        patterns = []
        pops = []
        for case, statements in reversed(self.cases.items()):
            mask = case
            if case is None:
                mask = '?' * len(self.test)
            if not statements:
                for pattern in patterns:
                    if all(p == c or p == '?' or c == '?' for p, c in zip(pattern, mask)):
                        break
                else:
                    pops.append(case)
            else:
                patterns.append(mask)

        for pop in pops:
            self.cases.pop(pop)

    @classmethod
    def convert_case(cls, test: ast.Value, case: int | str | tuple) -> list[str | None]:
        ret = []
        if isinstance(case, tuple):
            if len(case) == 0:
                case = ret.append(None)
            else:
                for c in case:
                    ret.extend(Switch.convert_case(test, c))
        elif isinstance(case, str):
            ret.append(case.replace('-', '?').replace(' ', ''))
        elif isinstance(case, int):
            ret.extend(cls.convert_case(test, format(case, f'0{len(test)}b')))
        else:
            raise RuntimeError(f"Unknown switch case: {case}")

        return ret

    @classmethod
    def process_cases(cls, test: ast.Value, cases: dict) -> dict[str | None, list[Statement]]:
        ret = {}

        for case, statements in cases.items():
            new_cases = cls.convert_case(test, case)

            for c in new_cases:
                ret[c] = statements

        return ret

    def _as_if(self) -> list[If | Else] | None:
        res = []

        for case, statements in self.cases.items():
            if case is None:
                if res:
                    res.append(self.Else(statements))
                else:
                    # FIX: Weird case, might not be filtered by previous stages
                    res.append(self.If(ast.Const(1, 1), statements))
                break

            if case.count('1') != 1 or not all(c in ['?', '1'] for c in case):
                return None

            bit = case[::-1].index('1')
            res.append(self.If(self.test[bit], statements))

        return res