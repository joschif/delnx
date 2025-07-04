"""Color palettes in addition to matplotlib's palettes."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import TYPE_CHECKING

from matplotlib import cm, colors

if TYPE_CHECKING:
    from collections.abc import Mapping

# --- Categorical Palettes ---


def get_categorical_palettes() -> Mapping[str, list[str]]:
    """Return a dictionary of default categorical color palettes."""
    # Colorblindness adjusted vega_10
    vega_10 = list(map(colors.to_hex, cm.tab10.colors))
    vega_10_scanpy = vega_10.copy()
    vega_10_scanpy[2] = "#279e68"  # green
    vega_10_scanpy[4] = "#aa40fc"  # purple
    vega_10_scanpy[8] = "#b5bd61"  # kakhi

    # default matplotlib 2.0 palette
    vega_20 = list(map(colors.to_hex, cm.tab20.colors))

    # reordered, some removed, some added
    vega_20_scanpy = [
        *vega_20[0:14:2],
        *vega_20[16::2],
        *vega_20[1:15:2],
        *vega_20[17::2],
        "#ad494a",
        "#8c6d31",
    ]
    vega_20_scanpy[2] = vega_10_scanpy[2]
    vega_20_scanpy[4] = vega_10_scanpy[4]
    vega_20_scanpy[7] = vega_10_scanpy[8]

    zeileis_28 = [
        "#023fa5",
        "#7d87b9",
        "#bec1d4",
        "#d6bcc0",
        "#bb7784",
        "#8e063b",
        "#4a6fe3",
        "#8595e1",
        "#b5bbe3",
        "#e6afb9",
        "#e07b91",
        "#d33f6a",
        "#11c638",
        "#8dd593",
        "#c6dec7",
        "#ead3c6",
        "#f0b98d",
        "#ef9708",
        "#0fcfc0",
        "#9cded6",
        "#d5eae7",
        "#f3e1eb",
        "#f6c4e1",
        "#f79cd4",
        "#7f7f7f",
        "#c7c7c7",
        "#1CE6FF",
        "#336600",
    ]

    godsnot_102 = [
        "#FFFF00",
        "#1CE6FF",
        "#FF34FF",
        "#FF4A46",
        "#008941",
        "#006FA6",
        "#A30059",
        "#FFDBE5",
        "#7A4900",
        "#0000A6",
        "#63FFAC",
        "#B79762",
        "#004D43",
        "#8FB0FF",
        "#997D87",
        "#5A0007",
        "#809693",
        "#6A3A4C",
        "#1B4400",
        "#4FC601",
        "#3B5DFF",
        "#4A3B53",
        "#FF2F80",
        "#61615A",
        "#BA0900",
        "#6B7900",
        "#00C2A0",
        "#FFAA92",
        "#FF90C9",
        "#B903AA",
        "#D16100",
        "#DDEFFF",
        "#000035",
        "#7B4F4B",
        "#A1C299",
        "#300018",
        "#0AA6D8",
        "#013349",
        "#00846F",
        "#372101",
        "#FFB500",
        "#C2FFED",
        "#A079BF",
        "#CC0744",
        "#C0B9B2",
        "#C2FF99",
        "#001E09",
        "#00489C",
        "#6F0062",
        "#0CBD66",
        "#EEC3FF",
        "#456D75",
        "#B77B68",
        "#7A87A1",
        "#788D66",
        "#885578",
        "#FAD09F",
        "#FF8A9A",
        "#D157A0",
        "#BEC459",
        "#456648",
        "#0086ED",
        "#886F4C",
        "#34362D",
        "#B4A8BD",
        "#00A6AA",
        "#452C2C",
        "#636375",
        "#A3C8C9",
        "#FF913F",
        "#938A81",
        "#575329",
        "#00FECF",
        "#B05B6F",
        "#8CD0FF",
        "#3B9700",
        "#04F757",
        "#C8A1A1",
        "#1E6E00",
        "#7900D7",
        "#A77500",
        "#6367A9",
        "#A05837",
        "#6B002C",
        "#772600",
        "#D790FF",
        "#9B9700",
        "#549E79",
        "#FFF69F",
        "#201625",
        "#72418F",
        "#BC23FF",
        "#99ADC0",
        "#3A2465",
        "#922329",
        "#5B4534",
        "#FDE8DC",
        "#404E55",
        "#0089A3",
        "#CB7E98",
        "#A4E804",
        "#324E72",
    ]

    return {
        "vega_10_scanpy": vega_10_scanpy,
        "vega_20_scanpy": vega_20_scanpy,
        "zeileis_28": zeileis_28,
        "godsnot_102": godsnot_102,
    }


# --- Continuous Palettes ---


def get_continuous_palettes() -> Mapping[str, colors.Colormap]:
    """Return a dictionary of default continuous color palettes (colormaps)."""
    # You can add more colormaps as needed
    return {
        "viridis": cm.get_cmap("viridis"),
        "plasma": cm.get_cmap("plasma"),
        "inferno": cm.get_cmap("inferno"),
        "magma": cm.get_cmap("magma"),
        "cividis": cm.get_cmap("cividis"),
        "Greys": cm.get_cmap("Greys"),
        "Blues": cm.get_cmap("Blues"),
        "Reds": cm.get_cmap("Reds"),
        "Greens": cm.get_cmap("Greens"),
    }


def default_palette(n: int) -> list[str]:
    """Return a categorical palette of length n, choosing the best palette based on n."""
    palettes = get_categorical_palettes()
    if n <= len(palettes["vega_10_scanpy"]):
        base = palettes["vega_10_scanpy"]
    elif n <= len(palettes["vega_20_scanpy"]):
        base = palettes["vega_20_scanpy"]
    elif n <= len(palettes["zeileis_28"]):
        base = palettes["zeileis_28"]
    else:
        base = palettes["godsnot_102"]
    if n <= len(base):
        return base[:n]
    # If n > base, repeat and cycle through palettes
    return list(itertools.islice(itertools.cycle(base), n))
