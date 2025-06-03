from importlib.metadata import version

from . import ds, pl, pp, tl

__all__ = ["pl", "pp", "tl", "ds"]

__version__ = version("delnx")
