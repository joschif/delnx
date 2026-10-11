"""Regenerate references/api.md from delnx's public signatures and docstrings.

Usage (from the repo root): python skills/delnx/scripts/build_api_md.py
"""

import inspect
from pathlib import Path

import delnx as dx

OUT = Path(__file__).resolve().parents[1] / "references" / "api.md"
MODULES = {"pp": dx.pp, "tl": dx.tl, "pl": dx.pl, "ds": dx.ds}


def public_functions(mod):
    """Public functions of a delnx submodule, in `__all__` order."""
    names = getattr(mod, "__all__", None) or [n for n in dir(mod) if not n.startswith("_")]
    return [(n, getattr(mod, n)) for n in names if inspect.isfunction(getattr(mod, n))]


def main() -> None:
    """Write references/api.md."""
    lines = [
        "# delnx API reference",
        "",
        "Generated from the delnx docstrings by `scripts/build_api_md.py` — do not edit by hand.",
        "",
    ]
    for short, mod in MODULES.items():
        lines += [f"## `dx.{short}`", ""]
        for name, fn in public_functions(mod):
            doc = inspect.getdoc(fn) or ""
            lines += [f"### `dx.{short}.{name}`", "", "```python", f"{name}{inspect.signature(fn)}", "```", "", doc, ""]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
