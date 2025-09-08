from celosia.hdl import HDL
from celosia.hdl.verilog.backend import VerilogModule

class Verilog(HDL):
    ModuleClass = VerilogModule

    top_first = False
    extensions = ['v']
    open_comment = '/* '
    close_comment = ' */'
