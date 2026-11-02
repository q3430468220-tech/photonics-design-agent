"""Shared fixtures: crossing builders for tests (numpy only)."""
import numpy as np

from prism_min.geometry import Geometry, Port, four_fold, profile_taper, rect, taper

W = 0.5


def ports(L):
    return [Port("W", (-L, 0), W, 180), Port("E", (L, 0), W, 0), Port("N", (0, L), W, 90), Port("S", (0, -L), W, 270)]


def plain(L=5.0):
    return Geometry(four_fold([rect(-L, -W / 2, 0, W / 2)]), ports(L))


def tapered(L=6.0, wc=1.5, Lt=3.0, straight=1.0, shape="linear"):
    x0, x1 = -L, -L + straight
    x2 = x1 + Lt
    arm = [rect(x0, -W / 2, x1 + 0.01, W / 2)]
    if shape == "linear":
        arm.append(taper(x1, W, x2, wc))
    else:
        xs = np.linspace(x1, x2, 80)
        ws = np.sqrt(W**2 + (wc**2 - W**2) * (xs - x1) / (x2 - x1))
        arm.append(profile_taper(xs, ws))
    arm.append(rect(x2 - 0.01, -wc / 2, 0, wc / 2))
    return Geometry(four_fold(arm), ports(L))


DESIGN_TEMPLATE = '''
import numpy as np
from prism_min.geometry import Geometry, Port, four_fold, rect, taper, profile_taper
W = 0.5
def build():
    L, wc, Lt, straight = {L}, {wc}, {Lt}, {straight}
    x0, x1 = -L, -L + straight
    x2 = x1 + Lt
    arm = [rect(x0, -W/2, x1 + 0.01, W/2)]
    if "{shape}" == "linear":
        arm.append(taper(x1, W, x2, wc))
    else:
        xs = np.linspace(x1, x2, 80)
        ws = np.sqrt(W**2 + (wc**2 - W**2) * (xs - x1) / (x2 - x1))
        arm.append(profile_taper(xs, ws))
    arm.append(rect(x2 - 0.01, -wc/2, 0, wc/2))
    ports = [Port("W", (-L, 0), W, 180), Port("E", (L, 0), W, 0), Port("N", (0, L), W, 90), Port("S", (0, -L), W, 270)]
    return Geometry(four_fold(arm), ports, thickness=0.22)
'''


def design_source(L, wc, Lt, straight=1.0, shape="linear"):
    return DESIGN_TEMPLATE.format(L=L, wc=wc, Lt=Lt, straight=straight, shape=shape)


def turn(hypothesis, change, filename, code, laws="none", expected="see change", change_type="parameter", label=""):
    return f"""### HYPOTHESIS
{hypothesis}

### CHANGE
{change}

### CHANGE TYPE
{change_type}

### LABEL
{label}

### EXPECTED
{expected}

### FILE {filename}
```python
{code}
```

### LAWS
{laws}
"""
