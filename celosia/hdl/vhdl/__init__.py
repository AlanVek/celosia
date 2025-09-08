from celosia.hdl import HDL
from celosia.hdl.vhdl.backend import VHDLModule

class VHDL(HDL):
    ModuleClass = VHDLModule

    top_first = False
    extensions = ['vhd', 'vhdl']
    open_comment = '-- '
    close_comment = ''
