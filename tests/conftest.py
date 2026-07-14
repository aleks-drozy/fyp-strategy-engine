import sys
import types

import pytest


@pytest.fixture
def fake_candle_module():
    """Register a fake top-level 'Candle' module with a 'Candle' class in
    sys.modules so the stdlib pickler can dump instances that round-trip
    exactly like the real Phase-6 source pickles (module==name=='Candle').
    RestrictedUnpickler never actually imports this module -- it substitutes
    its own local stub -- but the *dumping* side needs a real, importable
    module to satisfy pickle's save_global sanity check.
    """
    mod = types.ModuleType("Candle")

    class Candle:
        def __init__(self, o, h, l, c, t):
            self.o = o
            self.h = h
            self.l = l
            self.c = c
            self.t = t

    Candle.__module__ = "Candle"
    Candle.__qualname__ = "Candle"
    mod.Candle = Candle
    sys.modules["Candle"] = mod
    try:
        yield Candle
    finally:
        del sys.modules["Candle"]
