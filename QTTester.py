import ctypes, ctypes.util, sys, os
print("First QtCore    :", ctypes.util.find_library('Qt6Core'))

# force-load the DLL that lives next to the script, if any
ctypes.CDLL(os.path.abspath("Qt6Core.dll"), mode=ctypes.RTLD_GLOBAL)
print("Injected QtCore :", os.path.abspath("Qt6Core.dll"))

# now import PySide6 – this will die with your exact message
from PySide6 import QtCore