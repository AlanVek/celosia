from celosia.hdl import get_lang_map
from amaranth.build.plat import TemplatedPlatform

class Platform(TemplatedPlatform):
    @staticmethod
    def from_amaranth_platform(platform: TemplatedPlatform):
        if platform is not None:
            platform.toolchain_prepare = Platform.toolchain_prepare.__get__(platform)
        return platform

    def toolchain_prepare(self, fragment, name, *, emit_src=True, **kwargs):
        lang = kwargs.pop('lang', 'verilog')
        if not isinstance(lang, str):
            raise ValueError(f"Invalid 'lang': {lang}")

        ConverterClass = get_lang_map().get(lang.lower(), None)
        if ConverterClass is None:
            raise ValueError(f"Unknown 'lang': {lang}")

        converter = ConverterClass()
        try:
            converter.generate_overrides()
            plan = super().toolchain_prepare(fragment, name, emit_src=emit_src, **kwargs)
        finally:
            converter.cleanup_overrides()

        # TODO: Improve?
        verilog = get_lang_map()['verilog']
        for filename in (f'{name}', f'{name}.debug'):
            plan.files[f'{filename}.{converter.default_extension}'] = plan.files.pop(f'{filename}.v').replace(
                verilog.open_comment, converter.open_comment
            ).replace(
                verilog.close_comment, converter.close_comment
            )

        return plan