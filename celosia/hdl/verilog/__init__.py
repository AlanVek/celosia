from celosia.hdl.new_hdl import HDL
from celosia.hdl.verilog.backend import VerilogModule

class Verilog(HDL):
    ModuleClass = VerilogModule

    case_sensitive = True
    top_first = False
    extensions = ['v']
    open_comment = '/* '
    close_comment = ' */'
