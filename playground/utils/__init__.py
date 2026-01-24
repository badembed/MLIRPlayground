__all__ = ["memref"]

import importlib
import sys

def __getattr__(name):
    if name in __all__:
        submodule_path = f"{__name__}.{name}"
        submodule = importlib.import_module(submodule_path)
        setattr(sys.modules[__name__], name, submodule)
        return submodule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
